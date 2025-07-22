import rerun as rr
import argparse

# NOT WORKING

def main(rrd_file, port=9876):
    # Initialize rerun server (headless, no viewer)
    rr.init("Rerun_Logs_Viewer", spawn=True)
    rr.spawn(port=port)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.load(rrd_file)

    print(f"✅ Serving Rerun log '{rrd_file}' on port {port}")
    print(f"👉 Connect from local machine with:\n    rerun --connect rerun+http://127.0.0.1:{port}/proxy")

    # Keep running until manually killed
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped rerun server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve rerun .rrd logs via rerun SDK (no GUI).")
    parser.add_argument("rrd_file", type=str, help="Path to the .rrd log file.")
    parser.add_argument("--port", type=int, default=9876, help="Port to serve rerun viewer.")
    args = parser.parse_args()

    main(args.rrd_file, port=args.port)
