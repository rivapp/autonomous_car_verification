import re
import sys

def main(argv):

    input_filename = argv[0]

    totalTime = 0
    totalDnnTime = 0
    totalNumBranches = 0
    numRuns = 0
    
    with open(input_filename, 'r') as f:
        for line in f:
            m = re.search('seconds', line)

            if m is not None:

                items = line.split()
                curTime = items[3]
                curTime = float(curTime[0:len(curTime)-4])

                totalTime += curTime

            dnn = re.search('dnn', line)

            if dnn is not None:
                items = line.split()
                curTime = float(items[2])

                totalDnnTime += curTime

                numRuns += 1

            branch = re.search('branches', line)

            if branch is not None:
                items = line.split()
                curBranches = float(items[2])

                totalNumBranches += curBranches

    print('number of instances: ' + str(numRuns))
    print('average total runtime: ' + str(totalTime / numRuns))
    print('average NN runtime: ' + str(totalDnnTime / numRuns))
    print('average number of paths: ' + str(totalNumBranches / numRuns))

if __name__ == '__main__':
    main(sys.argv[1:])
