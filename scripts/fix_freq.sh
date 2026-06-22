#!/bin/sh
set -eu

detect_platform() {
    requested="${1:-auto}"
    case "$requested" in
        rk3576|rk3588|rk3588s)
            printf '%s\n' "$requested"
            return
            ;;
        auto|"") ;;
        *)
            echo "Unsupported TARGET_PLATFORM: $requested" >&2
            return 1
            ;;
    esac

    compatible="${RKLLM_DEVICE_COMPATIBLE:-}"
    if [ -z "$compatible" ]; then
        for path in \
            /proc/device-tree/compatible \
            /sys/firmware/devicetree/base/compatible \
            /proc/device-tree/model \
            /sys/firmware/devicetree/base/model
        do
            if [ -r "$path" ]; then
                compatible="$compatible $(tr '\000' ' ' < "$path")"
            fi
        done
    fi

    case "$compatible" in
        *rk3576*|*RK3576*) printf '%s\n' rk3576 ;;
        *rk3588*|*RK3588*) printf '%s\n' rk3588 ;;
        *)
            echo "Unable to detect RK3576 or RK3588 from the device tree." >&2
            echo "Set TARGET_PLATFORM explicitly." >&2
            return 1
            ;;
    esac
}

if [ "${1:-}" = "--detect" ]; then
    detect_platform "${2:-auto}"
    exit
fi

platform="$(detect_platform "${1:-${TARGET_PLATFORM:-auto}}")"
echo "Applying frequency settings for $platform"

case "$platform" in
    rk3576) exec /bin/sh /app/fix_freq_rk3576.sh ;;
    rk3588|rk3588s) exec /bin/sh /app/fix_freq_rk3588.sh ;;
esac

