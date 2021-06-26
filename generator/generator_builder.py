from generator.default_generator import DefaultGenerator
def get_generator(args):
    if args.dataset_type == 'default':
        train_generator = DefaultGenerator(args, mode="train",train_valid_split_ratio=args.train_valid_split_ratio,dataset_sample_ratio=args.dataset_sample_ratio)
        val_generator = DefaultGenerator(args, mode="valid",train_valid_split_ratio=args.train_valid_split_ratio,dataset_sample_ratio=args.dataset_sample_ratio)
    else:
        raise ValueError("{} is not supported!".format(args.dataset_type))
    return train_generator, val_generator