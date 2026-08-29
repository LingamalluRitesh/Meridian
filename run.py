#!/usr/bin/env python3
"""
ModelForge AI — Unified Enterprise Launcher
Coordinates asynchronous FastAPI backend server, MLOps orchestrator, and Next.js frontend.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

def parse_args():
    parser = argparse.ArgumentParser(description="ModelForge AI Unified Enterprise Launcher")
    parser.add_argument("--mode", choices=["all", "backend", "frontend", "test", "worker"], default="all",
                        help="Operating mode for ModelForge AI system")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Backend API server port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend UI port")
    return parser.parse_args()

def run_backend(host: str, port: int):
    print(f"[*] Launching ModelForge AI Backend Services on http://{host}:{port}...")
    import uvicorn
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    uvicorn.run("backend.api.main:app", host=host, port=port, log_level="info", reload=False)

def run_frontend(port: int):
    print(f"[*] Launching ModelForge AI Studio Dashboard on port {port}...")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    if os.path.exists(os.path.join(frontend_dir, "package.json")):
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)
    else:
        print("[!] Frontend package.json not found, running backend in headless mode.")

def run_tests():
    print("[*] Executing ModelForge AI Automated Test Suite...")
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    result = subprocess.run([sys.executable, "-m", "pytest", os.path.join(backend_dir, "tests"), "-v"])
    sys.exit(result.returncode)

def main():
    args = parse_args()
    print("=" * 70)
    print("⚡ MODELFORGE AI — ENTERPRISE AUTOMATED MACHINE LEARNING PLATFORM")
    print("=" * 70)
    
    if args.mode == "test":
        run_tests()
    elif args.mode == "backend":
        run_backend(args.host, args.port)
    elif args.mode == "frontend":
        run_frontend(args.frontend_port)
    elif args.mode == "all":
        backend_thread = threading.Thread(target=run_backend, args=(args.host, args.port), daemon=True)
        backend_thread.start()
        time.sleep(2)
        run_frontend(args.frontend_port)

if __name__ == "__main__":
    main()
