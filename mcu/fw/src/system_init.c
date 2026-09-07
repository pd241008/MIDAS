/* SystemInit: HSE 8 MHz → PLL 168 MHz (APB1=42MHz, APB2=84MHz)
 * FPU enabled in startup.S before this is called.
 */

#include "stm32f4xx.h"

/* PLL parameters for 168 MHz from 8 MHz HSE:
 *   PLL_M = 8   → VCO_in  = 8 MHz / 8   = 1 MHz
 *   PLL_N = 336 → VCO_out = 1 MHz × 336 = 336 MHz
 *   PLL_P = 2   → SYSCLK  = 336 / 2     = 168 MHz
 *   PLL_Q = 7   → USBCLK  = 336 / 7     = 48 MHz (unused but set correctly)
 */
#define PLL_M  8
#define PLL_N  336
#define PLL_P  2
#define PLL_Q  7

void SystemInit(void)
{
    /* Enable FPU (CP10/CP11 full access) — also done in startup, belt-and-suspenders */
    SCB_CPACR |= ((3UL << 20) | (3UL << 22));

    /* Enable DWT cycle counter (CYCCNT) */
    DEMCR |= (1UL << 24);   /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL   |= 1UL;      /* CYCCNTENA */

    /* Reset clock config to HSI (16 MHz internal) as safe starting point */
    RCC_CR |= RCC_CR_HSION;
    while (!(RCC_CR & (1UL << 1))) {} /* wait HSERDY just in case */

    /* Configure PLL: HSE source, M/N/P/Q */
    RCC_PLLCFGR = (PLL_M)
                | ((uint32_t)PLL_N << 6)
                | ((((uint32_t)PLL_P >> 1) - 1) << 16)
                | RCC_PLLCFGR_HSE
                | ((uint32_t)PLL_Q << 24);

    /* Enable HSE */
    RCC_CR |= RCC_CR_HSEON;
    while (!(RCC_CR & RCC_CR_HSERDY)) {}

    /* Set flash latency for 168 MHz: 5 wait states + prefetch + caches */
    FLASH_ACR = FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_5WS;

    /* AHB prescaler = 1, APB1 = /2 (42 MHz max), APB2 = /1 (84 MHz max) */
    RCC_CFGR = RCC_CFGR_PPRE1_DIV2;

    /* Enable PLL */
    RCC_CR |= RCC_CR_PLLON;
    while (!(RCC_CR & RCC_CR_PLLRDY)) {}

    /* Switch system clock to PLL */
    RCC_CFGR |= RCC_CFGR_SW_PLL;
    while ((RCC_CFGR & (0x3UL << 2)) != RCC_CFGR_SWS_PLL) {}

    /* Enable peripheral clocks: GPIOA (USART2 TX/RX), GPIOD (LED), USART2 */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIODEN | RCC_AHB1ENR_DMA2EN;
    RCC_APB1ENR |= RCC_APB1ENR_USART2EN;
}
