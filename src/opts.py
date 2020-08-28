
import argparse
class OPT:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='M2CT2020')

        # -------------------------------
        self.parser.add_argument('--wk_root', type=str, default='/home/roit/aws/aprojects/M2CT2020/proj')
        self.parser.add_argument('--data_dir',default='./data')
        self.parser.add_argument('--iteration',default=5000)


        self.parser.add_argument('--evaluate_freq',default=50)
        self.parser.add_argument('--out_dir',default='./out_dir')

        self.parser.add_argument('--data_out',default='./data_out.csv')

        self.parser.add_argument('--route_plt',
                                 default=[19,18,25,26,29,21,23,24,28,22,4,3,5,10,13,16,27,12,8,15,14,11,6,7
,9,2,1,0,17,20])
        #args
        self.parser.add_argument('--decay',default=0.9997)
        self.parser.add_argument('--learning_rate',default=0.9997)


        self.parser.add_argument('--routes',
                                 default=[
                                     [0,21,23,24,28,22,4,3],
                                     [0,5,13,27,16,10],
                                     [0,17,20,19,18,25,26,29],
                                     [0, 1, 9, 7, 6, 11, 14, 15,12,8,2]
                                 ])






    def args(self):
        self.options = self.parser.parse_args()
        return self.options

