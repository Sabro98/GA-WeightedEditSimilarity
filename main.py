from src.GeneticAlgorithm import Genetic
import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='dataset name(e, ic, gpcr, nr)', required=True)
    parser.add_argument('-dp', type=str, help='dataset path (default: data)', default='data')
    parser.add_argument('-rp', type=str, help='result path (default: result)', default='result')
    parser.add_argument('-pt', type=float, help='percentage of tournament of GA (default: 0.75)', default=0.75)
    parser.add_argument('-pm', type=float, help='percentage of mutate of GA (default: 0.025)', default=0.025)
    parser.add_argument('-rm', type=float, help='range of mutate of GA (default: 0.15)', default=0.15)
    parser.add_argument('-sz', type=int, help='population size of GA (default: 27)', default=27)
    parser.add_argument('-mg', type=int, help='max generation of GA (default: 2000)', default=2000)
    return parser

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    dataset = args.d
    datapath = args.dp
    resultpath = args.rp
    P_Mutate = args.pm
    P_Tournament = args.pt
    R_Mutate = args.rm
    sizeOfGroup = args.sz
    maxGeneration = args.mg

    if dataset not in ('e', 'ic', 'gpcr', 'nr'):
        parser.print_help()
        exit(0)

    # genetic method
    genetic = Genetic(dataset=dataset, datapath=datapath, resultpath=resultpath, P_Mutate=P_Mutate, P_Tournament=P_Tournament,
                        R_Mutate=R_Mutate, sizeOfGroup=sizeOfGroup, maxGeneration=maxGeneration)
    genetic.run()