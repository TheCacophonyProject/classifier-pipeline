import subprocess
import logging
import time

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
        cmd = "sudo systemctl restart thermal-postprocess"
    else:
        # disable but start once so that it can finish any stale files that may exist
        cmd = "sudo systemctl disable thermal-postprocess && sudo systemctl restart thermal-postprocess"
    return run_cmd(cmd)


def stop_network_classifier():
    cmd = "sudo systemctl stop thermal-classifier"
    return run_cmd(cmd)


def toggle_network_classifier(enable):
    if enable:
        cmd = "sudo systemctl start thermal-classifier"
    else:
        cmd = "sudo systemctl disable thermal-classifier && sudo systemctl stop thermal-classifier"
    return run_cmd(cmd)


def is_service_running(service_name):
    result = subprocess.run(["systemctl", "is-active", "--quiet", service_name])
    return result.returncode == 0

def preview_socket(headers, frame_queue):
    import yaml
    import socket
    from .signals import STOP_SIGNAL

    # convert casing
    python_dic = headers.__dict__
    go_dic = {}
    for k, v in python_dic.items():
        new_key = f"{k[0].upper()}{k[1:]}"
        try:
            under_index = new_key.index("_")
            new_key = f"{new_key[:under_index]}{new_key[under_index+1].upper()}{new_key[under_index+2:]}"
        except:
            pass
        go_dic[new_key] = v
    header_bytes = yaml.dump(go_dic).encode()
    header_bytes += b"\nclear"

    while True:
        try:
            # connect to management socket
            frameSocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            frameSocket.settimeout(5)
            frameSocket.connect("/var/spool/managementd")
            frameSocket.settimeout(None)
            logging.info("Connected to management interface")
            frameSocket.send(header_bytes)
            telemetry_bytes = bytearray(640)
            # if we need this can add the correct info
            while True:
                frame = frame_queue.get()
                if isinstance(frame, str):
                    if frame == STOP_SIGNAL:
                        return
                if frame is None:
                    logging.info("Disconnected")
                    break
                frame_bytes = frame.pix.byteswap().tobytes()
                frame_bytes = telemetry_bytes + frame_bytes
                frameSocket.send(frame_bytes)
        except:
            logging.error("Failed to connect to /var/spool/managementd", exc_info=True)
            try:
                # empty the queue
                items = frame_queue.qsize()
                items = max(items, 1)
                for _ in range(items):
                    item = frame_queue.get(100)
                    if isinstance(item, str):
                        if item == STOP_SIGNAL:
                            return
            except:
                pass
            # could not connect wait a few seconds
            time.sleep(2)





def kill_process_with_timeout(process, timeout=30):
    from threading import Thread
    # for some reason process.kill hangs sometimes
    kill_thread = Thread(target=kill_process, args=(process,),daemon=True)
    kill_thread.start()
    try:
        kill_thread.join(timeout)
        if kill_thread.is_alive():
            logging.error("Kill thread didn't terminate, should terminate when parent process terminates")
    except:
        logging.error("Kill thread didnt terminate", exc_info=True)


def kill_process(process):
    import psutil
    pid = process.pid
    logging.info("Killing process %s", pid)
    try:
        parent = psutil.Process(pid)
        # 1. Filter out Zombies and Uninterruptible Sleep (D-state) processes instantly
        try:
            status = parent.status()
            if status == psutil.STATUS_ZOMBIE:
                logging.info("PID %s is a Zombie; skipping signal.", pid)
                return
            if status == 'uninterruptible-sleep': # D-state
                logging.warning("PID %s is stuck in D-state (I/O hang). kill -9 will fail.", pid)
        except psutil.NoSuchProcess:
            return
        
        children = parent.children()
        for child in children:
            if child.is_running():
                kill_process(child)
        psutil.wait_procs(children, timeout=5)
        if parent.is_running():
            try:
                parent.kill()
                logging.info("Killed %s",process.pid)
            except:
                logging.error("Could not kill process %s ", parent.pid, exc_info=True)
            parent.wait(5)
    except:
        logging.error("Could not kill process", exc_info=True)