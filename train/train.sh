#!/bin/bash
python train.py 2>&1 | tee train.log
print_warn "60秒后自动关机，如需取消，请立即执行："
echo -e "${BOLD}sudo shutdown -c${NC}"
sleep 60
/usr/bin/shutdown