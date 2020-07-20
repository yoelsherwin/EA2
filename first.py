from __future__ import print_function
import numpy as np
import create_params
import os
import neat
import visualize

FILENAME = "train.csv"
SIZE = 3000

data = create_params.special_normalize(FILENAME, SIZE)

inputs = np.zeros(shape=(SIZE, 16))
outputs = np.zeros(shape=(SIZE))

idx = 0
for line in data:
    a,b,c,d = create_params.create_hyper_params(line[1:])
    all = np.zeros(shape=(16))
    for i in range(4):
        all[i] = a[i]
        all[i + 4] = b[i]
        all[i + 8] = c[i]
        all[i + 12] = d[i]
    inputs[idx] = all
    outputs[idx] = line[0]
    idx += 1


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i, o in zip(inputs, outputs):
            output = net.activate(i)
            if output[0] <= .5:
                ans = 0
            else:
                ans = 1
            if ans == o == 1:
                TP += 1
            elif ans == 1 and o == 0:
                FP += 1
            elif ans == o == 0:
                TN += 1
            else:
                FN += 1
        if ((TP + FP) == 0):
            precision = 0
        else:
            precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if ((TP + FN) == 0):
            recall = 0
        else:
            recall = TP / (TP + FN)
        if ((0.0625 * precision + recall) != 0):
            # Fbeta = (1.015625 * precision * recall) / (0.015625 * precision + recall)
            Fbeta = (1.0625 * precision * recall) / (0.0625 * precision + recall)
        else:
            Fbeta = 0
        genome.fitness = Fbeta


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        if output <= .5:
            ans = 0
        else:
            ans = 1
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, ans))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)