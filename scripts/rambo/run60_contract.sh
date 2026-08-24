#!/usr/bin/env bash
# Shared pre-Kit argument contract for the dedicated RAMBO simulator wrappers.
#
# This file is sourced by ``run60.sh`` and ``run_runtime_artifact.sh``.  It is
# deliberately shell-only: rejecting an unsafe invocation must not import
# Python, Isaac Lab, Kit, or a physics backend.

rambo_contract_error() {
    echo "RAMBO launch contract violation: $1" >&2
}

rambo_validate_launch_contract() {
    local -a arguments=("$@")
    local argument option_name option_name_lower viz_value
    local index=0
    local viz_count=0

    while (( index < ${#arguments[@]} )); do
        argument="${arguments[index]}"
        case "${argument}" in
            --viz)
                if (( index + 1 >= ${#arguments[@]} )); then
                    rambo_contract_error "--viz requires exactly one value: none or kit"
                    return 2
                fi
                ((index += 1))
                viz_value="${arguments[index]}"
                ;;
            --viz=*)
                viz_value="${argument#--viz=}"
                ;;
            --visualizer|--visualizer=*)
                rambo_contract_error "--visualizer is forbidden; use exactly one --viz none or --viz kit"
                return 2
                ;;
            --headless|--headless=*)
                rambo_contract_error "--headless is forbidden; use --viz none"
                return 2
                ;;
            --experience|--experience=*|--kit_args|--kit_args=*|--kit-args|--kit-args=*)
                rambo_contract_error "unaudited Kit experience arguments are forbidden: ${argument%%=*}"
                return 2
                ;;
            *)
                viz_value=""
                ;;
        esac

        if [[ "${argument}" == --viz || "${argument}" == --viz=* ]]; then
            if [[ "${viz_value}" != "none" && "${viz_value}" != "kit" ]]; then
                rambo_contract_error "--viz only accepts none or kit"
                return 2
            fi
            ((viz_count += 1))
        fi

        # No CLI argument is allowed to select or override the fixed PhysX
        # configuration.  Match option names only, never ordinary positional
        # values such as paths or artifact labels that happen to mention a
        # dependency package.
        if [[ "${argument}" == --* ]]; then
            option_name="${argument%%=*}"
            option_name_lower="${option_name,,}"
            case "${option_name_lower}" in
                *newton*|*physics*|*physx*|--backend|--backend-*|--backend_*|--sim-backend|--sim_backend)
                    rambo_contract_error "physics/backend override is forbidden: ${option_name}"
                    return 2
                    ;;
            esac
        fi

        ((index += 1))
    done

    if (( viz_count != 1 )); then
        rambo_contract_error "exactly one explicit --viz none or --viz kit is required"
        return 2
    fi
}
