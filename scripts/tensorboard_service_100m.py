"""Run a foreground metrics bridge and localhost TensorBoard together."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"work/pretrain_100m_2b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=6006)
    parser.add_argument("--runs",type=Path,default=BASE/"runs")
    parser.add_argument("--logdir",type=Path,default=BASE/"tensorboard")
    args = parser.parse_args()
    with socket.socket() as s:
        try: s.bind(("127.0.0.1",args.port))
        except OSError: raise SystemExit(f"Port {args.port} is already occupied. Use the existing dashboard or choose --port.")
    args.logdir.mkdir(parents=True,exist_ok=True)
    commands = [
        [sys.executable,"-u",str(ROOT/"scripts/tensorboard_metrics_100m.py"),"--runs",str(args.runs),"--logdir",str(args.logdir)],
        [sys.executable,"-m","tensorboard.main","--logdir",str(args.logdir),"--host","127.0.0.1",
         "--port",str(args.port),"--reload_interval","2","--load_fast=false","--purge_orphaned_data=true"]]
    children, stopping = [], [False]
    def stop(signum,frame): stopping[0] = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    state_path = BASE/"tensorboard_service.json"
    state = {"pid":os.getpid(),"port":args.port,"logdir":str(args.logdir),
             "url":f"http://127.0.0.1:{args.port}/","training_started_by_service":False}
    try:
        for command in commands: children.append(subprocess.Popen(command,cwd=ROOT,start_new_session=True))
        state["child_pids"] = [c.pid for c in children]
        state["status"] = "starting"
        state_path.write_text(json.dumps(state,indent=2)+"\n")
        ready = False
        while not stopping[0]:
            for child in children:
                if child.poll() is not None: raise RuntimeError(f"Dashboard child exited {child.returncode}")
            if not ready:
                try:
                    with urllib.request.urlopen(state["url"]+"data/environment",timeout=1) as response:
                        ready = response.status == 200
                except OSError: pass
                if ready:
                    state["status"]="ready"
                    state_path.write_text(json.dumps(state,indent=2)+"\n")
                    print(f"TensorBoard READY: {state['url']}\nJupyterLab: replace /lab... in the current browser URL with /proxy/{args.port}/\nScalars: train/loss, validation/loss, validation/perplexity. Auto-reload: 5 seconds in TensorBoard Settings.\nNo formal training logs yet means an empty dashboard; training appears automatically. Ctrl+C stops only the dashboard.",flush=True)
            time.sleep(1)
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill();child.wait()
        state["status"]="stopped"
        state_path.write_text(json.dumps(state,indent=2)+"\n")


if __name__ == "__main__": main()
