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

/* ---- Constants ---- */
static constexpr int kInputDim   = 12;
static constexpr int kArenaSize  = 16 * 1024;  /* 16 KB tensor arena */
static constexpr int kNumIters   = 5;           /* repeat each model for stable timing */

/* Item 6 (DMA ring buffer): host streams feature vectors over USART2 RX.
 * DMA2 Stream5 Channel4 dumps bytes into a 256 B circular buffer; the
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

/* End of a host frame: push vector into the window, quantize latest into g_input. */
static void finalize_vector(void)
{
    if (g_vec_field != kInputDim - 1) { return; }
    for (int i = 0; i < kInputDim; i++) {
        g_win[g_win_cnt % kWindowLen][i] = g_vec_tmp[i];
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

/* DMA2 Stream5 Channel4 = USART2_RX, circular into g_dma_rx. */
static void dma_uart_rx_init(void)
{
    /* Reset the stream (EN=0, wait for hardware to clear) */
    DMA2_Stream5->CR = 0;
    while (DMA2_Stream5->CR & 1u) {}
    DMA2_Stream5->CR = (4u << 25)      /* CHSEL=4 (USART2_RX) */
                     | (1u << 10)      /* MINC  */
                     | (1u << 8);      /* CIRC  */
    DMA2_Stream5->NDTR = (uint16_t)kRingBufSize;
    DMA2_Stream5->PAR  = (uint32_t)&USART2->DR;
    DMA2_Stream5->M0AR = (uint32_t)g_dma_rx;
    DMA2_Stream5->CR  |= 1u;           /* EN */

    /* USART2: connect RX to DMA and frame on IDLE line breaks */
    USART2->CR3 |= USART_CR3_DMAR;
    USART2->CR1 |= USART_CR1_IDLEIE;
}

/* Frame boundary: DMA owns byte reads; pause DMAR while clearing IDLE (HAL-safe). */
extern "C" void USART2_IRQHandler(void)
{
    if (USART2->SR & USART_SR_IDLE) {
        USART2->CR3 &= ~USART_CR3_DMAR;
        (void)USART2->SR;
        (void)USART2->DR;
        USART2->CR3 |= USART_CR3_DMAR;

        uint16_t tail = (uint16_t)(kRingBufSize - (DMA2_Stream5->NDTR & 0xFFFF));
        uint16_t avail = (uint16_t)((tail - g_dma_head + kRingBufSize) % kRingBufSize);
        if (avail >= (uint16_t)kRingBufSize) { avail = (uint16_t)(kRingBufSize - 1); }
        if (avail > 0) { consume_rx_bytes(g_dma_rx, avail); }
    }
}

/* Sample NSL input (dummy probe values — real pipeline fills from ring buffer) */
static void make_dummy_input(void)
{
    /* Placeholder: uniform [0,1] features → int8 quantization.
     * In production, the host fills g_input via ring buffer at d=12. */
    static const float dummy[kInputDim] = {
        0.5f, 0.3f, 0.1f, 0.7f, 0.2f, 0.4f,
        0.6f, 0.8f, 0.3f, 0.9f, 0.1f, 0.5f
    };
    for (int i = 0; i < kInputDim; i++) {
        g_input[i] = quantize_feature(dummy[i]);
    }
}

static int run_model(tflite::MicroInterpreter *interp, const char *name)
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

    /* Dequantize output: prob = (int8 - zp) * scale */
    int8_t raw_out = output->data.int8[0];
    float prob = ((float)raw_out - (float)kOutputZp) * kOutputScale;

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
    uart2_puts("Running inference x");
    uart2_put_u32(kNumIters);
    uart2_puts(" each model...\n");

    /* Fill dummy input features (real pipeline: ring buffer at d=12) */
    make_dummy_input();

    /* ---- Warm up + benchmark both models ---- */
    for (int iter = 0; iter < kNumIters; iter++) {
        led_on(iter % 2 ? 1 : 3);  /* alternate orange/blue per iteration */

        uart2_puts("--- iter ");
        uart2_put_u32(iter + 1);
        uart2_puts(" ---\n");

        /* NSL specialist (provenance=0 routing path) */
        run_model(g_nsl_interp, "NSL ");

        /* UNSW specialist (provenance=1 routing path) */
        run_model(g_unsw_interp, "UNSW");

        led_off(1);
        led_off(3);
    }

    uart2_puts("=== DONE ===\n");
    led_on(0);  /* green = finished */

    /* ---- Production ingest (Item 6): DMA ring buffer live ---- */
    uart2_puts("=== PROD: DMA RX live. Host: 12 comma-separated floats + LF ===\n");
    while (1) {
        if (g_new_frame) {
            g_new_frame = 0;
            led_on((g_win_cnt & 1u) ? 1 : 3);
            run_model(g_nsl_interp, "NSL ");
            run_model(g_unsw_interp, "UNSW");
            led_off(1);
            led_off(3);
        }
        __asm volatile("wfi");
    }
    return 0;
}
