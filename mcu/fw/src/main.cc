/* STM32F407VG bare-metal TFLM inference firmware
 * Runs two per-source QAT INT8 specialists (NSL + UNSW) on shared 12-dim overlap features.
 * Outputs cycle counts + quantized probabilities via USART2 @ 115200 8N1.
 *
 * Memory layout:
 *   Tensor arena  = 16 KB  (both models fit: ~8 KB each; reused sequentially)
 *   Stack         = 1 KB  (in remaining SRAM)
 *   TFLM library  = ~20 KB code + ~2 KB .data/.bss  (from libtensorflow-microlite.a)
 *   Models        = ~5 KB flash each (two C arrays)
 *   Total SRAM    ~20 KB peak, well under 192 KB budget
 *   Total flash   ~30 KB, well under 1 MB budget
 */

#include "stm32f4xx.h"
#include "models/model_nsl.h"
#include "models/model_unsw.h"

/* TFLM headers */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <math.h>

/* ---- Constants ---- */
static constexpr int kInputDim   = 12;
static constexpr int kArenaSize  = 16 * 1024;  /* 16 KB tensor arena */
static constexpr int kNumIters   = 50;          /* repeat each model for stable timing */

/* Item 6 (DMA ring buffer): host streams feature vectors over USART2 RX.
 * DMA1 Stream5 Channel4 dumps bytes into a 256 B circular buffer; the
 * IDLE-line IRQ frames each complete vector ("f1,...,f12\n") and shifts it
 * into a W-deep float window for the defense, quantizing the latest vector
 * into g_input for inference.
 */
static constexpr int   kRingBufSize = 256;   /* DMA RX circular byte buffer  */
static constexpr int   kWindowLen   = 10;    /* W feature vectors in window  */

/* NSL quantization params (from tflite introspection):
 *   input  scale=0.003921569 zp=-128  (x/255 - 128, input domain [0,255])
 *   output scale=1.0/256   zp=0      (fixed-point [0,1) mapped to uint8)
 */
static constexpr float kInputScale  = 0.003921569f;
static constexpr int   kInputZp     = -128;
static constexpr float kOutputScale = 1.0f / 256.0f;
static constexpr int   kOutputZp    = 0;

/* ---- On-device routing gate (folded ridge-logistic on the 12-dim vector).
 * Host math:  z=(x-mu)/sd;  logit = z@W + b;  prob = sigmoid(logit).
 * Fold standardization into the weights offline:  logit = x@An + bn, an
 * identical linear decision (bit-match verified: sign agree 1.0). Route =
 * UNSW specialist iff logit>0 (prob>0.5).
 */
static const float g_An[kInputDim] = {
    -6.263695f, -18.52529f, 3.271755f, 0.2495766f, -5.014591f, 25.18609f,
    1.354753f, -3.461511f, -3.501773f, -4.4492f, 5.026102f, 6.561226f
};
static const float g_bn = 3.756836f;

/* Sigmoid via a 64-entry LUT over logit in [-8, +8] (flash table). */
static const float g_sigmoid_lut[64] = {
    0.000335f, 0.000432f, 0.000557f, 0.000718f, 0.000926f, 0.001193f,
    0.001537f, 0.001981f, 0.002552f, 0.003288f, 0.004234f, 0.005452f,
    0.007017f, 0.009027f, 0.011607f, 0.014913f, 0.019143f, 0.024542f,
    0.031414f, 0.040133f, 0.051143f, 0.064969f, 0.082209f, 0.103518f,
    0.129570f, 0.161002f, 0.198320f, 0.241796f, 0.291339f, 0.346396f,
    0.405897f, 0.468297f, 0.531703f, 0.594103f, 0.653604f, 0.708661f,
    0.758204f, 0.801680f, 0.838998f, 0.870430f, 0.896482f, 0.917791f,
    0.935031f, 0.948857f, 0.959867f, 0.968586f, 0.975458f, 0.980857f,
    0.985087f, 0.988393f, 0.990973f, 0.992983f, 0.994548f, 0.995766f,
    0.996712f, 0.997448f, 0.998019f, 0.998463f, 0.998807f, 0.999074f,
    0.999282f, 0.999443f, 0.999568f, 0.999665f
};

/* ---- Fix-basis manifold rotation (rotation.rs port).
 * b0 = normalized manifold centroid, b1 = Gram-Schmidt e1 (offline);
 * Givens rotation of x in the (b0,b1) plane by delta_theta.
 */
static const float g_b0[kInputDim] = {
    0.002763292f, 0.2465388f, 0.1439028f, 0.8243486f, 2.281197e-05f,
    1.837985e-05f, 0.01430074f, 0.1115384f, 0.04933199f, 0.4387532f,
    0.08957263f, 0.1527931f
};
static const float g_b1[kInputDim] = {
    -0.0007029569f, 0.9691329f, -0.03660759f, -0.209707f, -5.803164e-06f,
    -4.675671e-06f, -0.003637983f, -0.02837438f, -0.01254962f, -0.1116149f,
    -0.02278648f, -0.03886922f
};

/* Defense config (MidasConfig defaults, host-verified): gamma=0.292,
 * lambda=1.0, k=2.0, delta_theta_max=45 deg = 0.785398 rad. */
static constexpr float kGamma   = 0.292f;
static constexpr float kLambda  = 1.0f;
static constexpr float kK       = 2.0f;
static constexpr float kThetaMax = 0.785398f;

/* ---- UART2 bare-metal I/O ---- */
static void uart2_init(void)
{
    /* PA2 = AF7 (USART2 TX), PA3 = AF7 (USART2 RX) */
    GPIOA->MODER &= ~((3UL << 4) | (3UL << 6));
    GPIOA->MODER |=  ((2UL << 4) | (2UL << 6));   /* AF mode */
    GPIOA->AFR[0] &= ~((0xFUL << 8) | (0xFUL << 12));
    GPIOA->AFR[0] |=  ((7UL << 8)  | (7UL << 12));  /* AF7 */

    /* BRR for 115200 @ 42 MHz APB1: 42000000/115200 ≈ 364.58 → mantissa=36, frac=10 */
    USART2->BRR = (36 << 4) | 10;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

static void uart2_putc(char c)
{
    while (!(USART2->SR & USART_SR_TXE)) {}
    USART2->DR = (uint32_t)c;
}

static void uart2_puts(const char *s)
{
    while (*s) { uart2_putc(*s++); }
}

static void uart2_put_u32(uint32_t v)
{
    char buf[11];
    int i = 10;
    buf[i] = '\0';
    if (v == 0) { uart2_putc('0'); return; }
    while (v > 0) { buf[--i] = '0' + (v % 10); v /= 10; }
    uart2_puts(&buf[i]);
}

static void uart2_put_float(float f)
{
    if (f < 0.0f) { uart2_putc('-'); f = -f; }
    uint32_t whole = (uint32_t)f;
    uart2_put_u32(whole);
    uart2_putc('.');
    uint32_t frac = (uint32_t)((f - (float)whole) * 10000.0f);
    char buf[5]; int i = 4; buf[4] = '\0';
    for (int j = 0; j < 4; j++) { buf[--i] = '0' + (frac % 10); frac /= 10; }
    uart2_puts(buf);
}

/* ---- LED helpers (PD12=green, PD13=orange, PD14=red, PD15=blue) ---- */
static void led_init(void)
{
    GPIOD->MODER &= ~((3UL << 24) | (3UL << 26) | (3UL << 28) | (3UL << 30));
    GPIOD->MODER |=  ((1UL << 24) | (1UL << 26) | (1UL << 28) | (1UL << 30));
}

static void led_on(int n)  { GPIOD->BSRR = (1UL << (12 + n)); }
static void led_off(int n) { GPIOD->BSRR = (1UL << (12 + n + 16)); }

/* ---- TFLM inference ---- */
static tflite::MicroInterpreter *g_nsl_interp = nullptr;
static tflite::MicroInterpreter *g_unsw_interp = nullptr;
static uint8_t g_arena[kArenaSize];

/* ---- DMA RX ring buffer + vector parser state (Item 6) ---- */
static uint8_t  g_dma_rx[kRingBufSize];
static volatile uint16_t g_dma_head = 0;      /* bytes already consumed      */
static float    g_win[kWindowLen][kInputDim]; /* trailing W feature vectors  */
static uint32_t g_win_cnt = 0;                /* vectors ingested (total)    */
static volatile int  g_new_frame = 0;         /* full window + new frame     */
static int8_t   g_input[kInputDim];           /* quantized current vector    */

static float    g_vec_tmp[kInputDim];
static int      g_vec_field = 0;
static char     g_num[32];
static int      g_num_len = 0;

static float g_x[kInputDim];  /* latest parsed [0,1] vector (gate + rotate input) */

static int run_model(tflite::MicroInterpreter *interp, const char *name,
                     volatile uint32_t *cyc_out, int idx, int cap);  /* defined below */

/* ---- Per-stage DWT capture (defense pipeline trace) ---- */
static volatile uint32_t g_stage_cyc[5][kNumIters]; /* 0=sense 1=theta 2=gate 3=rotate 4=inf */
static volatile uint8_t  g_route[kNumIters];        /* routed specialist: 0=NSL 1=UNSW */
static volatile uint32_t g_bench_done = 0;
static int g_quiet = 0;  /* suppress per-invoke UART chatter during trace */

/* Compact fixed-point float parser ("-1.2345") for the host's %f output. */
static float parse_float(const char *s, int len)
{
    int i = 0;
    float neg = 1.0f;
    if (i < len && s[i] == '-') { neg = -1.0f; i++; }
    float v = 0.0f;
    while (i < len && s[i] >= '0' && s[i] <= '9') { v = v * 10.0f + (float)(s[i] - '0'); i++; }
    if (i < len && s[i] == '.') {
        i++;
        float frac = 0.1f;
        while (i < len && s[i] >= '0' && s[i] <= '9') {
            v += (float)(s[i] - '0') * frac; frac *= 0.1f; i++;
        }
    }
    return neg * v;
}

/* Quantize a [0,1] normalized feature to int8 (scale=1/255, zp=-128). */
static int8_t quantize_feature(float x)
{
    int q = (int)(x * 255.0f + 0.5f) + kInputZp;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    return (int8_t)q;
}

/* Field terminator: commit the current decimal into the partial vector. */
static void commit_field(void)
{
    if (g_num_len > 0 && g_vec_field < kInputDim) {
        g_vec_tmp[g_vec_field] = parse_float(g_num, g_num_len);
    }
    g_num_len = 0;
}

/* End of a host frame: push vector into the window, snapshot it for the
 * gate/rotation, quantize latest into g_input. */
static void finalize_vector(void)
{
    if (g_vec_field != kInputDim - 1) { return; }
    for (int i = 0; i < kInputDim; i++) {
        g_win[g_win_cnt % kWindowLen][i] = g_vec_tmp[i];
        g_x[i] = g_vec_tmp[i];
        g_input[i] = quantize_feature(g_vec_tmp[i]);
    }
    g_win_cnt++;
    if (g_win_cnt >= (uint32_t)kWindowLen) { g_new_frame = 1; }
}

static void parse_rx_char(char c)
{
    if (c == ',' || c == '\n' || c == '\r') {
        commit_field();
        if (c == ',') {
            g_vec_field++;
        } else {
            if (g_vec_field == kInputDim - 1) { finalize_vector(); }
            g_vec_field = 0;
        }
    } else if (g_num_len < (int)sizeof(g_num) - 1) {
        g_num[g_num_len++] = c;
    }
}

static void consume_rx_bytes(const uint8_t *buf, uint16_t n)
{
    for (uint16_t k = 0; k < n; k++) {
        parse_rx_char((char)buf[(g_dma_head + k) % kRingBufSize]);
    }
    g_dma_head = (uint16_t)((g_dma_head + n) % kRingBufSize);
}

/* DMA1 Stream5 Channel4 = USART2_RX, circular into g_dma_rx. */
static void dma_uart_rx_init(void)
{
    /* Reset the stream (EN=0, wait for hardware to clear) */
    DMA1_Stream5->CR = 0;
    while (DMA1_Stream5->CR & 1u) {}
    DMA1_Stream5->CR = (4u << 25)      /* CHSEL=4 (USART2_RX) */
                     | (1u << 10)      /* MINC  */
                     | (1u << 8);      /* CIRC  */
    DMA1_Stream5->NDTR = (uint16_t)kRingBufSize;
    DMA1_Stream5->PAR  = (uint32_t)&USART2->DR;
    DMA1_Stream5->M0AR = (uint32_t)g_dma_rx;
    DMA1_Stream5->CR  |= 1u;           /* EN */

    /* USART2: connect RX to DMA and frame on IDLE line breaks */
    USART2->CR3 |= USART_CR3_DMAR;
    USART2->CR1 |= USART_CR1_IDLEIE;
}

/* Frame boundary: DMA owns byte reads; pause DMAR while clearing IDLE (HAL-safe).
 * Extracted into a callable so the trace benchmark can measure the full
 * ingest path inside a DWT window (identical code path as the ISR). */
static void ingest_idle_frame(void)
{
    USART2->CR3 &= ~USART_CR3_DMAR;
    (void)USART2->SR;
    (void)USART2->DR;
    USART2->CR3 |= USART_CR3_DMAR;

    uint16_t tail = (uint16_t)(kRingBufSize - (DMA1_Stream5->NDTR & 0xFFFF));
    uint16_t avail = (uint16_t)((tail - g_dma_head + kRingBufSize) % kRingBufSize);
    if (avail >= (uint16_t)kRingBufSize) { avail = (uint16_t)(kRingBufSize - 1); }
    if (avail > 0) { consume_rx_bytes(g_dma_rx, avail); }
}

extern "C" void USART2_IRQHandler(void)
{
    if (USART2->SR & USART_SR_IDLE) { ingest_idle_frame(); }
}

/* Trace mode: synthesize a host ASCII line into the DMA ring, then let
 * ingest_idle_frame() consume it exactly as a real IDLE event would.
 * NDTR must be programmed with the stream disabled (RM0433: writing NDTR
 * while EN=1 is ignored). */
static void simulate_rx_line(const char *line)
{
    int len = 0;
    while (line[len]) { len++; }
    g_dma_head = 0;
    if (len < kRingBufSize) {
        uint32_t cr = DMA1_Stream5->CR;
        DMA1_Stream5->CR = cr & ~1u;          /* EN=0 */
        while (DMA1_Stream5->CR & 1u) {}
        DMA1_Stream5->NDTR = (uint16_t)(kRingBufSize - len);
        DMA1_Stream5->CR = cr | 1u;           /* restore EN */
        for (int i = 0; i < len; i++) { g_dma_rx[i] = (uint8_t)line[i]; }
    }
}

/* ---- Defense pipeline (gate + theta + fixed-basis rotation) ---- */

/* cos of the angle between consecutive vectors (momentum M_i in trajectory.rs). */
static float compute_momentum(const float *a, const float *b)
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < kInputDim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float den = sqrtf(na * nb);
    if (den < 1e-7f) { return 0.0f; }
    float m = dot / den;
    if (m >  1.0f) { m =  1.0f; }
    if (m < -1.0f) { m = -1.0f; }
    return m;
}

/* Windowed penetration: eps = (1/|W|) * sum_i ||v_{i+1}||2 * max(0, M_i). */
static float __attribute__((noinline)) penetration_epsilon_windowed(void)
{
    if (g_win_cnt < 2) { return 0.0f; }
    float sum = 0.0f;
    int pairs = (int)g_win_cnt - 1;
    if (pairs > kWindowLen - 1) { pairs = kWindowLen - 1; }
    for (int i = 0; i < pairs; i++) {
        const float *cur  = g_win[(g_win_cnt - kWindowLen + i + 1) % kWindowLen];
        const float *prev = g_win[(g_win_cnt - kWindowLen + i)     % kWindowLen];
        float m = compute_momentum(cur, prev);
        if (m < 0.0f) { m = 0.0f; }
        float no = 0.0f;
        for (int j = 0; j < kInputDim; j++) { no += cur[j] * cur[j]; }
        sum += sqrtf(no) * m;
    }
    return sum / (float)pairs;
}

/* Delta-theta budget (rotation.rs): eps<=gamma linear, else exponential,
 * capped at delta_theta_max (45 deg). */
static float __attribute__((noinline)) rotation_angle(float eps)
{
    float th = (eps <= kGamma) ? kLambda * eps
                               : kLambda * expf(kK * (eps - kGamma));
    if (th > kThetaMax) { th = kThetaMax; }
    return th;
}

/* Folded gate logit (bit-match of the host ridge-logistic, standardization
 * folded into g_An/g_bn offline). noinline: keeps the work between the
 * caller's DWT counter reads (GCC may otherwise sink pure inline code past
 * the trailing volatile counter read, collapsing the measured window). */
static float __attribute__((noinline)) gate_logit(const float *x)
{
    float s = g_bn;
    for (int i = 0; i < kInputDim; i++) { s += x[i] * g_An[i]; }
    return s;
}

/* sigmoid via 64-entry LUT over logit in [-8, +8]. */
static float __attribute__((noinline)) sigmoid_lut(float logit)
{
    int idx = (int)((logit + 8.0f) * (64.0f / 16.0f));
    if (idx < 0)  { idx = 0; }
    if (idx > 63) { idx = 63; }
    return g_sigmoid_lut[idx];
}

/* Givens rotation in the fixed (b0,b1) basis plane by theta (rotation.rs). */
static void __attribute__((noinline)) rotate_givens(float theta, const float *x, float *out)
{
    float c = cosf(theta), s = sinf(theta);
    float c0 = 0.0f, c1 = 0.0f;
    for (int i = 0; i < kInputDim; i++) {
        c0 += x[i] * g_b0[i];
        c1 += x[i] * g_b1[i];
    }
    float r0 = c * c0 - s * c1;
    float r1 = s * c0 + c * c1;
    for (int i = 0; i < kInputDim; i++) {
        out[i] = x[i] + (r0 - c0) * g_b0[i] + (r1 - c1) * g_b1[i];
    }
}

/* Full per-frame defense step, DWT-windowed per stage when store>=0:
 *  stage[0] T_sense : IDLE-frame ingest (DMA pokes + ASCII parse + window + quantize)
 *  stage[1] T_theta : windowed penetration + rotation angle
 *  stage[2] T_gate  : 12x1 FC + logistic LUT + route
 *  stage[3] T_rotate: fixed-basis Givens rotation
 *  stage[4] T_inf   : quantize rotated vector + routed specialist Invoke
 * Returns the routed specialist (0=NSL, 1=UNSW). */
static int step_defense_pipeline(int store, int idx, int do_ingest)
{
    uint32_t a0, a1;

    if (do_ingest) {
        a0 = DWT_CYCCNT;
        ingest_idle_frame();
        a1 = DWT_CYCCNT;
        if (store >= 0) { g_stage_cyc[0][idx] = a1 - a0; }
    }

    a0 = DWT_CYCCNT;
    float eps = penetration_epsilon_windowed();
    float theta = rotation_angle(eps);
    a1 = DWT_CYCCNT;
    if (store >= 0) { g_stage_cyc[1][idx] = a1 - a0; }

    a0 = DWT_CYCCNT;
    float logit = gate_logit(g_x);
    (void)sigmoid_lut(logit);   /* included in the T_gate window */
    int route = (logit > 0.0f) ? 1 : 0;
    a1 = DWT_CYCCNT;
    if (store >= 0) { g_stage_cyc[2][idx] = a1 - a0; g_route[idx] = (uint8_t)route; }

    static float x_rot[kInputDim];
    a0 = DWT_CYCCNT;
    rotate_givens(theta, g_x, x_rot);
    a1 = DWT_CYCCNT;
    if (store >= 0) { g_stage_cyc[3][idx] = a1 - a0; }

    for (int i = 0; i < kInputDim; i++) {
        g_input[i] = quantize_feature(x_rot[i]);
    }
    tflite::MicroInterpreter *interp = route ? g_unsw_interp : g_nsl_interp;
    a0 = DWT_CYCCNT;
    run_model(interp, route ? "UNSW" : "NSL ", nullptr, -1, 0);
    a1 = DWT_CYCCNT;
    if (store >= 0) { g_stage_cyc[4][idx] = a1 - a0; }

    return route;
}

/* Trace benchmark: cycle 4 real feature seeds through the full ingest +
 * defense + routed-invoke pipeline, recording per-stage DWT cycles. */
static void trace_defense_pipeline(void)
{
    static const char *seeds[4] = {
        "0.1906,0.3333,0.0000,0.7500,0.0000,1.0000,0.0000,0.0020,0.0020,1.0000,0.0169,0.0000\n", /* gate→UNSW  */
        "0.0000,0.3333,0.0444,0.5000,0.0000,0.0000,0.0000,0.0039,0.0039,0.5882,0.0029,0.0027\n", /* gate→NSL   */
        "0.0000,0.3333,0.1111,0.2500,0.0000,0.0000,1.0000,0.0020,0.0020,0.0039,0.0169,0.0159\n", /* gate→NSL   */
        "0.0000,0.0149,0.0000,0.6875,0.0000,0.0000,0.0000,0.0020,0.0020,0.1882,0.0169,0.0159\n"  /* gate→UNSW  */
    };

    uart2_puts("Defense pipeline trace, iters=");
    uart2_put_u32(kNumIters);
    uart2_puts(" (4 real feature seeds)...\n");

    g_quiet = 1;
    for (int iter = 0; iter < kNumIters; iter++) {
        simulate_rx_line(seeds[iter & 3]);
        step_defense_pipeline(1, iter, 1);
    }
    g_quiet = 0;

    g_bench_done = 1;
    uart2_puts("=== DONE defense trace ===\n");
}

static int run_model(tflite::MicroInterpreter *interp, const char *name,
                     volatile uint32_t *cyc_out, int idx, int cap)
{
    TfLiteTensor *input  = interp->input_tensor(0);
    TfLiteTensor *output = interp->output_tensor(0);
    if (!input || !output) { uart2_puts("ERR:TENSOR\n"); return -1; }

    /* Fill input (copy pre-quantized int8 values) */
    for (int i = 0; i < kInputDim; i++) {
        input->data.int8[i] = g_input[i];
    }

    uint32_t start = DWT_CYCCNT;
    TfLiteStatus st = interp->Invoke();
    uint32_t cycles = DWT_CYCCNT - start;

    if (st != kTfLiteOk) { uart2_puts("ERR:INVOKE\n"); return -1; }

    if (cyc_out && idx >= 0 && idx < cap) { cyc_out[idx] = cycles; }

    /* Dequantize output: prob = (int8 - zp) * scale */
    int8_t raw_out = output->data.int8[0];
    float prob = ((float)raw_out - (float)kOutputZp) * kOutputScale;

    if (g_quiet) { return 0; }

    uart2_puts(name);
    uart2_putc(':');
    uart2_puts("cyc=");
    uart2_put_u32(cycles);
    uart2_puts(" prob=");
    uart2_put_float(prob);
    uart2_putc('\n');

    return 0;
}

/* ---- init_model ---- */
static tflite::MicroInterpreter *init_model(const unsigned char *model_buf,
                                            size_t model_len,
                                            const char *label)
{
    (void)model_len;
    const tflite::Model *model = tflite::GetModel(model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart2_puts(label); uart2_puts(":ERR:SCHEMA\n");
        return nullptr;
    }

    /* Register only the ops used by these 3-layer specialists:
     *   FullyConnected, Logistic (sigmoid), Tanh
     */
    static tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddTanh();

    /* Use the shared arena for both models (init sequentially, not simultaneously) */
    static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        uart2_puts(label); uart2_puts(":ERR:ALLOC\n");
        return nullptr;
    }
    return &interp;
}

/* ---- main ---- */
extern "C" int main(void)
{
    /* Init peripherals */
    uart2_init();
    dma_uart_rx_init();
    /* USART2 IRQn = 38 → NVIC_ISER[1] bit 6 */
    NVIC_ISER[1] |= (1u << 6);
    led_init();
    led_on(0);  /* green LED on = boot */

    uart2_puts("\n=== STM32F407VG TFLM specialist firmware ===\n");

    /* ---- Init NSL specialist ---- */
    uart2_puts("Loading NSL model... ");
    g_nsl_interp = init_model(model_nsl_data, model_nsl_len, "NSL");
    uart2_puts(g_nsl_interp ? "OK\n" : "FAIL\n");

    /* ---- Init UNSW specialist ---- */
    uart2_puts("Loading UNSW model... ");
    g_unsw_interp = init_model(model_unsw_data, model_unsw_len, "UNSW");
    uart2_puts(g_unsw_interp ? "OK\n" : "FAIL\n");

    if (!g_nsl_interp || !g_unsw_interp) {
        uart2_puts("FATAL: Model init failed. Halting.\n");
        while (1) { led_on(2); }  /* red LED = error */
    }

    /* Show arena usage */
    uart2_puts("Arena used: ");
    uart2_put_u32(g_nsl_interp->arena_used_bytes());
    uart2_puts(" bytes (NSL) / ");
    uart2_put_u32(g_unsw_interp->arena_used_bytes());
    uart2_puts(" bytes (UNSW)\n");

    led_off(0);
    uart2_puts("Running defense pipeline trace x");
    uart2_put_u32(kNumIters);
    uart2_puts(" frames...\n");

    /* Real feature seeds; fill the window a few rounds first (steady state) */
    trace_defense_pipeline();

    uart2_puts("=== DONE ===\n");
    led_on(0);  /* green = finished */

    /* ---- Production ingest (Item 6): DMA ring live + full defense pipeline ---- */
    uart2_puts("=== PROD: DMA RX live. Host: 12 comma-separated floats + LF ===\n");
    while (1) {
        if (g_new_frame) {
            g_new_frame = 0;
            led_on((g_win_cnt & 1u) ? 1 : 3);
            g_quiet = 0;
            step_defense_pipeline(-1, 0, 0);
            led_off(1);
            led_off(3);
        }
        __asm volatile("wfi");
    }
    return 0;
}
