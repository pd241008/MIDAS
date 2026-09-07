/* STM32F407VGT6 linker script
 * Flash: 1024 KB @ 0x08000000
 * SRAM:  128 KB  @ 0x20000000  (main work area)
 * CCM:    64 KB  @ 0x10000000  (data-only, not DMA-capable)
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
    CCM   (rwx) : ORIGIN = 0x10000000, LENGTH = 64K
}

_estack = ORIGIN(SRAM) + LENGTH(SRAM);

SECTIONS
{
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        *(.rodata)
        *(.rodata*)
        *(.glue_7)
        *(.glue_7t)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT> FLASH

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > SRAM

    /* _end / end = symbol for sbrk heap start (nosys.specs) */
    PROVIDE(end = _ebss);
    PROVIDE(__dso_handle = 0);

    /* Tensor arena goes into SRAM, aligned to 16 bytes */
    .tensor_arena (NOLOAD) :
    {
        . = ALIGN(16);
        *(.tensor_arena)
        . = ALIGN(16);
    } > SRAM

    /* Discard debug/comment sections to save flash */
    /DISCARD/ :
    {
        *(.ARM.attributes)
    }
}
