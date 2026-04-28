#!/bin/bash
# 激活 CAN 接口，并按传感器类型检查触觉设备节点
set -e

BITRATE=1000000
SENSOR_TYPE="${1:-none}"

usage() {
    echo "Usage: bash interface_up.sh [8chips|4chips|crab|none]"
}

recover_ch340() {
    local iface=""
    local cur_driver=""

    # Stop common services that may grab CH340 via usbfs.
    sudo systemctl stop ModemManager brltty brltty-udev.service >/dev/null 2>&1 || true
    sudo pkill brltty >/dev/null 2>&1 || true

    sudo modprobe usbserial >/dev/null 2>&1 || true
    sudo modprobe ch341 >/dev/null 2>&1 || true

    for n in /sys/bus/usb/devices/*; do
        [[ -f "$n/idVendor" ]] || continue
        local vid pid
        vid=$(cat "$n/idVendor" 2>/dev/null || true)
        pid=$(cat "$n/idProduct" 2>/dev/null || true)
        if [[ "$vid:$pid" == "1a86:7523" ]]; then
            for i in "$n":*; do
                [[ -d "$i" ]] || continue
                iface=$(basename "$i")
                break 2
            done
        fi
    done

    if [[ -z "$iface" ]]; then
        echo "Warn: CH340 interface not found"
        return 1
    fi

    if [[ -L "/sys/bus/usb/devices/$iface/driver" ]]; then
        cur_driver=$(basename "$(readlink -f "/sys/bus/usb/devices/$iface/driver")")
        if [[ -e "/sys/bus/usb/drivers/$cur_driver/unbind" ]]; then
            echo "$iface" | sudo tee "/sys/bus/usb/drivers/$cur_driver/unbind" >/dev/null || true
        fi
    fi

    echo "$iface" | sudo tee /sys/bus/usb/drivers/ch341/bind >/dev/null || true
    sudo udevadm control --reload-rules >/dev/null 2>&1 || true
    sudo udevadm trigger --subsystem-match=tty >/dev/null 2>&1 || true
}

bring_up_can() {
    for iface in can_master can_follower can_master2 can_follower2; do
        if ! ip link show "$iface" &>/dev/null; then
            echo "Warn: $iface not found. Skipping..."
            continue
        fi
        sudo ip link set "$iface" down 2>/dev/null || true
        sudo ip link set "$iface" type can bitrate "$BITRATE"
        sudo ip link set "$iface" up
        echo "$iface activated at ${BITRATE} bps"
    done
}

ensure_node() {
    local node="$1"
    if [[ ! -e "$node" ]]; then
        echo "Warn: required device node not found: $node"
        recover_ch340 || true
    fi

    if [[ ! -e "$node" ]]; then
        echo "Error: required device node not found: $node"
        echo "Hint: reconnect device, or run the CH340 quick-fix commands in README."
        exit 1
    fi
}

check_tactile() {
    case "$SENSOR_TYPE" in
        8chips)
            ensure_node /dev/tactile_8chips
            echo "tactile sensor ready: /dev/tactile_8chips"
            ;;
        4chips)
            ensure_node /dev/tactile_4chips_left
            ensure_node /dev/tactile_4chips_right
            echo "tactile sensor ready: /dev/tactile_4chips_left + /dev/tactile_4chips_right"
            ;;
        crab)
            ensure_node /dev/tactile_sensor
            echo "tactile sensor ready: /dev/tactile_sensor"
            ;;
        none)
            echo "tactile sensor check skipped"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

bring_up_can
check_tactile

echo "interface setup done (sensor_type=$SENSOR_TYPE)"
