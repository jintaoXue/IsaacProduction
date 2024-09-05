#!/bin/bash
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MY_DIR="$(realpath -s "$BASE_DIR")"

SCRIPT_DIR="${HOME}/.local/share/ov/pkg/isaac-sim-2023.1.1"

# path=$SCRIPT_DIR
# while [[ $path != / ]];
# do
    
#     if ! find "$path" -maxdepth 1 -mindepth 1 -iname "_build" -exec false {} +
#     then
#         break
#     fi
#     # Note: if you want to ignore symlinks, use "$(realpath -s "$path"/..)"
#     path="$(readlink -f "$path"/..)"
    
# done
# build_path=$path/_build
export CARB_APP_PATH=$SCRIPT_DIR/kit
export EXP_PATH=$SCRIPT_DIR/apps
export ISAAC_PATH=$SCRIPT_DIR
# Show icon if not running headless
export RESOURCE_NAME="IsaacSim"
# WAR for missing libcarb.so
export LD_PRELOAD=$SCRIPT_DIR/kit/libcarb.so
# echo ${MY_DIR} $SCRIPT_DIR
. ${MY_DIR}/isaac_setup_python_env.sh
