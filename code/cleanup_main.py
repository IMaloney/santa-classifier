from scripts.cleanup import parse_args, remove_folders, delete_run


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        remove_folders()
    elif args.run > 0:
        delete_run(args.run)