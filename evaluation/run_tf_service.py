"""
Launches a tf.data.service worker and dispatcher on this machine.
"""

import tensorflow as tf
import time
import argparse

DISPATCHER_PORT = 38655
WORKER_PORT = 38656


def main():
    parser = argparse.ArgumentParser(description="Server for tf.data.service")
    parser.add_argument(
        "--ip_addr",
        type=str,
        help="IP Address of local host",
        required=True,
    )
    args = parser.parse_args()

    d_config = tf.data.experimental.service.DispatcherConfig(
        port=DISPATCHER_PORT
    )
    dispatcher = tf.data.experimental.service.DispatchServer(d_config)
    dispatcher_address = dispatcher.target.split("://")[1]

    print("Started tf.data service at address {}".format(dispatcher.target))

    w_config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=dispatcher_address,
        worker_address=args.ip_addr + ":" + str(WORKER_PORT),
        port=WORKER_PORT,
    )
    worker = tf.data.experimental.service.WorkerServer(w_config)  # noqa:F841

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
