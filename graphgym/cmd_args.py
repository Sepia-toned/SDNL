# import argparse
# import sys


# def parse_args():
#     """Parses the arguments."""
#     parser = argparse.ArgumentParser(
#         description='Train a classification model'
#     )
#     parser.add_argument(
#         '--cfg',
#         dest='cfg_file',
#         help='Config file path',
#         required=False,
#         type=str,
#         default="/home/lidaqiu/DLJF_service_16/roland/roland-master/run/roland_example.yaml"
#     )
#     parser.add_argument(
#         '--repeat',
#         dest='repeat',
#         help='Repeat how many random seeds',
#         default=1,
#         type=int
#     )
#     parser.add_argument(
#         '--mark_done',
#         dest='mark_done',
#         action='store_true',
#         help='mark yaml as yaml_done after a job has finished',
#     )

#     parser.add_argument(
#         '--override_remark',
#         dest='override_remark',
#         type=str,
#         required=False,
#         default="roland_example",
#         help='easily override the remark in the yaml file'
#     )

#     parser.add_argument(
#         '--override_data_dir',
#         dest='override_data_dir',
#         type=str,
#         required=False,
#         default="/home/lidaqiu/DLJF_service_16/roland/roland-master/roland_public_data",
#         help='easily override the dataset.dir in the yaml file'
#     )

#     parser.add_argument(
#         'opts',
#         help='See graphgym/config.py for all options',
#         default=None,
#         nargs=argparse.REMAINDER
#     )

#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)
#     return parser.parse_args()

import argparse
import sys


def parse_args():
    """Parse command line args for training entry scripts."""
    parser = argparse.ArgumentParser(
        description="Train a classification model"
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        required=True,  # 强制显式传入配置文件，避免误用历史默认路径
        help="Config file path",
    )

    parser.add_argument(
        "--repeat",
        dest="repeat",
        type=int,
        default=1,
        help="Repeat how many random seeds",
    )

    parser.add_argument(
        "--mark_done",
        dest="mark_done",
        action="store_true",
        help="Mark yaml as *_done after a job has finished",
    )

    # 关键改动：默认 None，仅显式传入时才覆盖 yaml
    parser.add_argument(
        "--override_remark",
        dest="override_remark",
        type=str,
        default=None,
        help="Override cfg.remark only when explicitly provided",
    )

    # 关键改动：默认 None，仅显式传入时才覆盖 yaml
    parser.add_argument(
        "--override_data_dir",
        dest="override_data_dir",
        type=str,
        default=None,
        help="Override cfg.dataset.dir only when explicitly provided",
    )

    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional KEY VALUE pairs to override config",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()
