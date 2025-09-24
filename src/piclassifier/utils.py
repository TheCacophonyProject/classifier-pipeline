import subprocess
import logging


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            encoding="ascii",
            check=True,
        )
        return result.returncode == 0

    except:
        logging.error("Could not run command %s", cmd, exc_info=True)
        return False


def startup_postprocessor(enable):
    if enable:
        cmd = "sudo systemctl enable thermal-postprocess && sudo systemctl restart thermal-postprocess"
    else:
        # disable but start once so that it can finish any stale files that may exist
        cmd = "sudo systemctl disable thermal-postprocess && sudo systemctl restart thermal-postprocess"
    return run_cmd(cmd)


def stop_network_classifier():
    cmd = "sudo systemctl stop thermal-classifier"
    return run_cmd(cmd)


def startup_network_classifier(enable):
    if enable:
        cmd = "sudo systemctl enable thermal-classifier && sudo systemctl restart thermal-classifier"
    else:
        cmd = "sudo systemctl disable thermal-classifier && sudo systemctl stop thermal-classifier"
    return run_cmd(cmd)


def is_service_running(service_name):
    result = subprocess.run(["systemctl", "is-active", "--quiet", service_name])
    return result.returncode == 0
