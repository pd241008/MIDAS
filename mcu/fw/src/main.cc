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

/* Sample NSL input (dummy probe values — real pipeline fills from ring buffer) */
static int8_t g_input[kInputDim];

static void make_dummy_input(void)
{
    /* Placeholder: uniform [0,1] features → int8 quantization.
     * In production, the host fills g_input via ring buffer at d=12. */
    static const float dummy[kInputDim] = {
        0.5f, 0.3f, 0.1f, 0.7f, 0.2f, 0.4f,
        0.6f, 0.8f, 0.3f, 0.9f, 0.1f, 0.5f
    };
    for (int i = 0; i < kInputDim; i++) {
        int q = (int)(dummy[i] * 255.0f + 0.5f) + kInputZp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        g_input[i] = (int8_t)q;
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

    while (1) {
        __asm volatile("wfi");
    }
    return 0;
}
