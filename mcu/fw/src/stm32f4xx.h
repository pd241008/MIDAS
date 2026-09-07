/* Minimal STM32F407 register definitions for bare-metal TFLM firmware.
 * Only the registers used: RCC, FLASH, GPIOx, USART2, SysTick, SCB, FPU CPACR.
 */

#ifndef STM32F4XX_H
#define STM32F4XX_H

#include <stdint.h>

/* ---- Cortex-M4 system registers ---- */
#define SCS_BASE        0xE000E000UL
#define SysTick         ((volatile uint32_t *)(SCS_BASE + 0x0010UL))
#define NVIC_ICPR       ((volatile uint32_t *)(SCS_BASE + 0x0180UL))
#define NVIC_ISER       ((volatile uint32_t *)(SCS_BASE + 0x0100UL))
#define SCB             ((volatile uint32_t *)(SCS_BASE + 0x0D00UL))
#define SCB_CPACR       (*(volatile uint32_t *)(SCS_BASE + 0x0888UL))  /* CPACR */

/* DWT cycle counter (for profiling) */
#define DWT             ((volatile uint32_t *)(SCS_BASE + 0x0100UL))
#define DWT_CYCCNT      (*(volatile uint32_t *)(SCS_BASE + 0x0104UL))
#define DWT_CTRL        (*(volatile uint32_t *)(SCS_BASE + 0x0000UL))
#define DEMCR           (*(volatile uint32_t *)(SCS_BASE + 0x0000UL + 0x00FCUL))

/* ---- Peripheral base addresses ---- */
#define PERIPH_BASE     0x40000000UL
#define APB1_BASE       PERIPH_BASE
#define AHB1_BASE       (PERIPH_BASE + 0x00020000UL)

/* RCC */
#define RCC_BASE        (AHB1_BASE + 0x3800UL)
#define RCC_CR          (*(volatile uint32_t *)(RCC_BASE + 0x00))
#define RCC_PLLCFGR     (*(volatile uint32_t *)(RCC_BASE + 0x04))
#define RCC_CFGR        (*(volatile uint32_t *)(RCC_BASE + 0x08))
#define RCC_AHB1ENR     (*(volatile uint32_t *)(RCC_BASE + 0x30))
#define RCC_APB1ENR     (*(volatile uint32_t *)(RCC_BASE + 0x40))

/* FLASH */
#define FLASH_BASE      (AHB1_BASE + 0x3C00UL)
#define FLASH_ACR       (*(volatile uint32_t *)(FLASH_BASE + 0x00))

/* GPIO */
#define GPIOA_BASE      (AHB1_BASE + 0x0000UL)
#define GPIOB_BASE      (AHB1_BASE + 0x0400UL)
#define GPIOC_BASE      (AHB1_BASE + 0x0800UL)
#define GPIOD_BASE      (AHB1_BASE + 0x0C00UL)

typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

#define GPIOA   ((GPIO_TypeDef *)GPIOA_BASE)
#define GPIOD   ((GPIO_TypeDef *)GPIOD_BASE)

/* USART2 (on APB1) */
#define USART2_BASE     (APB1_BASE + 0x4400UL)
typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_TypeDef;

#define USART2  ((USART_TypeDef *)USART2_BASE)

/* DMA2 (AHB1, base 0x40026000) — USART2_RX = DMA2 Stream5 Channel4 */
typedef struct {
    volatile uint32_t CR;
    volatile uint32_t NDTR;
    volatile uint32_t PAR;
    volatile uint32_t M0AR;
    volatile uint32_t M1AR;
    volatile uint32_t FCR;
} DMA_Stream_TypeDef;

#define DMA2_BASE       (AHB1_BASE + 0x6000UL)
#define DMA2_Stream5    ((DMA_Stream_TypeDef *)(DMA2_BASE + 0x00A0UL))

/* RCC_CR bits */
#define RCC_CR_HSION   (1UL << 0)
#define RCC_CR_HSIRDY  (1UL << 1)
#define RCC_CR_HSEON   (1UL << 16)
#define RCC_CR_HSERDY  (1UL << 17)
#define RCC_CR_PLLON   (1UL << 24)
#define RCC_CR_PLLRDY  (1UL << 25)

/* RCC_CFGR bits */
#define RCC_CFGR_SW_PLL        (0x2UL << 0)
#define RCC_CFGR_SWS_PLL       (0x2UL << 2)
#define RCC_CFGR_PPRE1_DIV2    (0x4UL << 10)

/* RCC_PLLCFGR bits */
#define RCC_PLLCFGR_HSE        (1UL << 22)
#define RCC_PLLCFGR_SRC_HSE    RCC_PLLCFGR_HSE

/* RCC_AHB1ENR bits */
#define RCC_AHB1ENR_GPIOAEN    (1UL << 0)
#define RCC_AHB1ENR_GPIODEN    (1UL << 3)
#define RCC_AHB1ENR_DMA2EN     (1UL << 22)

/* RCC_APB1ENR bits */
#define RCC_APB1ENR_USART2EN   (1UL << 17)

/* FLASH_ACR bits (for 168 MHz = 5 wait states) */
#define FLASH_ACR_LATENCY_5WS  0x05UL
#define FLASH_ACR_ICEN         (1UL << 9)
#define FLASH_ACR_DCEN         (1UL << 10)
#define FLASH_ACR_PRFTEN       (1UL << 8)

/* USART_CR1 bits */
#define USART_CR1_UE           (1UL << 13)
#define USART_CR1_TE           (1UL << 3)
#define USART_CR1_RE           (1UL << 2)
#define USART_CR1_IDLEIE       (1UL << 4)

/* USART_CR3 bits */
#define USART_CR3_DMAR         (1UL << 6)

/* USART_SR bits */
#define USART_SR_TXE           (1UL << 7)
#define USART_SR_TC            (1UL << 6)
#define USART_SR_RXNE          (1UL << 5)
#define USART_SR_IDLE          (1UL << 4)

#endif /* STM32F4XX_H */
