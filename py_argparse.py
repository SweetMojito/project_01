import argparse


def main():
    # 构造参数解析器
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # 添加参数
    parser.add_argument("--name", type=str, default="World")
    parser.add_argument("--dayno", type=int, default=20240808)

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    print(f"接收到的参数: {args}")
    print(f"接收到的参数: {args.name}")
    print(f"接收到的参数: {args.dayno}")


if __name__ == "__main__":
    main()
