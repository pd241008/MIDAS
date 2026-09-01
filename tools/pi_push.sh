#!/bin/bash
# Push a local file to the Raspberry Pi (avina@192.168.137.43) through the
# Windows/OpenSSH bridge. pi.sh cannot copy files and the bridge mangles
# pipes/quotes/parens; base64 is alphanumeric + '/ + '=' so it survives cleanly.
#
# Usage:
#   tools/pi_push.sh <local-path> <remote-abs-path>
set -e

KEY='C:\Users\prath\.ssh\midas_key'
PI_USER=avina
PI_HOST=192.168.137.43

LOCAL="${1:?usage: pi_push.sh <local> <remote-abs-path>}"
REMOTE="${2:?usage: pi_push.sh <local> <remote-abs-path>}"

# base64 the file content (single line, url-safe-ish; standard base64 works)
B64=$(base64 -w0 "$LOCAL")

# One ssh call that decodes stdin and writes to the remote path. base64 string is
# embedded directly (no shell metacharacters besides / + = which survive).
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  "& ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no $PI_USER@$PI_HOST 'echo $B64 | base64 -d > $REMOTE; wc -c $REMOTE' 2>&1 | Out-String" \
  2>&1 | tr -d '\r'
echo "pi_push: $LOCAL -> $REMOTE"
