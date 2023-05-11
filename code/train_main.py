from model.runner import test_and_train, parse_args


if __name__ == "__main__":
    args = parse_args()
    run_number, is_transfer_model, gcp = args.run_number, args.transfer_learning, args.gcp
    test_and_train(run_number, is_transfer_model, gcp)