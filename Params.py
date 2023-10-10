import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    # Dataset Part
    parser.add_argument('--dataset', default='Ciao', help='dataset path: \dataset\Epinions\Ciao')

    # Multi-behaviors Part
    parser.add_argument('--layers', type=int, default=2, help='number of MB-Soc layers')
    parser.add_argument('--alpha', type=float, default=0.2, help='rate of leakyReLU')
    parser.add_argument('--behav_num', type=int, default=3, help='total number of behaviors')
    parser.add_argument('--embed_dim', type=int, default=32, help='dimension of the embedding')
    parser.add_argument('--tau', type=float, default=0.1, help='tau in CL_Loss')
    parser.add_argument('--behav_loss_rate', type=float, default=0.5, help='behavior loss rate in CL_Loss')
    
    # Train and Test Part
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--test_u_batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='p of dropout layer')
    parser.add_argument('--reg_rate', type=float,default=5e-4, help="the rate of l2 regularization")
    parser.add_argument('--cl_rate', type=float,default=1e-3, help="contrast learning rate")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate in fine-tuning')
    parser.add_argument('--lr_dc', type=float, default=0.7, help='decay rate of the learning rate ')
    parser.add_argument('--lr_dc_step', type=int, default=20, help='the number of steps after which the learning rate decay')
    parser.add_argument('--test_rate', type=float, default='0.1', help='the rate of dataset for test')
    parser.add_argument('--early_steps', type=int, default=2, help='steps of the early-stop learning strategy[Test every 5 epochs]')

    # Pre-Train
    parser.add_argument('--pre_epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--pre_reg_rate', type=float,default=5e-4, help="the rate of l2 regularization")
    parser.add_argument('--pre_cl_rate', type=float,default=1e-3, help="contrast learning rate")
    parser.add_argument('--pre_lr', type=float, default=1e-3, help='learning rate in pre-training')

    # Model Part
    parser.add_argument('--load_model', default=None, help='the model name to load')
    parser.add_argument('--model_name', default='MBSoc', help='the name for save [MBSoc]')

    return parser.parse_args()


args = parse_args()
