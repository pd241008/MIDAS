#!/bin/bash
# SSH to the Raspberry Pi 5 (avina@192.168.137.43) through the Windows host,
# which is the only path that reaches the 192.168.137.x hotspot the Pi joined.
#
# NOTE: the Pi has NO working RTC battery and its hotspot NTP path is unreliable,
# so its clock drifts back to 1970 on every reboot. A wrong clock silently breaks
# TLS (curl/cargo/git certs). Use:  tools/pi.sh sync   to set it from this box
# (which has the correct time) before any network-dependent operation.
#
# Usage:
#   tools/pi.sh sync                 re-sync the Pi clock to this machine's time
#   tools/pi.sh 'command string'     run a command on the Pi (single quotes only)
#                                    AVOID internal pipes/quotes in the command —
#                                    they get mangled by the Windows bridge.
#
# Key: Windows C:\Users\username\.ssh\midas_key  (public key on Pi's authorized_keys)
set -e

PI_USER=avina
PI_HOST=192.168.137.43
PI_SSH="ssh -i C:\Users\prath\.ssh\midas_key -o BatchMode=yes -o StrictHostKeyChecking=no $PI_USER@$PI_HOST"

bridge() {
  /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
    "& $PI_SSH '$*' 2>&1 | Out-String" \
    2>&1 | tr -d '\r'
}

# Capture the exact epoch at call time (correct clock on this dev box).
pi_epoch=$(date +%s)

# Use a single SSH call that sets the clock then runs nothing (idempotent sync).
do_sync() {
  /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
    "& $PI_SSH 'echo pd12345678 | sudo -S date -s @$pi_epoch 2>&1 >/dev/null; date -u' 2>&1 | Out-String" \
    2>&1 | tr -d '\r'
}

if [ "$1" = "sync" ]; then
  do_sync
  exit 0
fi

bridge "$*"
