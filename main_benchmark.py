#!/usr/bin/env python

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from shutil import rmtree
import sys

# np.set_printoptions(threshold=np.inf)

from circuit_searcher import CircuitSearcher


def fluxonium(N, E_c, E_j, phi, n):
    """
    Hamiltonium function computing fluxonium qubit eigenspectrum. 
    :param N: Number of array junctions
    :param E_c: Capacitor energies
    :param E_j: Junction energies
    :param phi: flux sweep to be used
    :param n: Level of energy 
    :return: eigen spectum of N junctions
    """
    gamma = 1.5
    E_l = (gamma * E_j) / N
    4 * E_c * (n ** 2) - E_j * np.cos(phi) + 0.5 * E_l * (phi ** 2)


def plot_basic_spec(spectrum, plot_title: str, plot_savefig: str, phi_ext):
    """
    :param spectrum: numpy array of eigenspectrums computed directly from circuit searcher.
    :param plot_title: Title of the plot
    :param plot_savefig: Title of the savefig
    :return: None
    """
    plt.figure()
    plt.scatter(x=phi_ext, y=spectrum[:, 0], label="Ground State")
    plt.scatter(x=phi_ext, y=spectrum[:, 1], label='First Excited State')
    plt.scatter(x=phi_ext, y=spectrum[:, 2], label='Second Excited State')
    plt.xlabel("Phi-External")
    plt.ylabel("Energy (GHz)")
    plt.title(plot_title)
    plt.savefig(plot_savefig)
    return None


if __name__ == '__main__':
    """
        Set parameters and run the inverse design algorithm.
        general_params:
            solver: string         | specifies circuit solver - 'JJcircuitSim' or '2-node'
            phiExt: array		   | external fluxes for which to solve circuit
            target_spectrum: array | target flux spectrum of circuit (used by specific loss functions only)
        Note: Task names are assumed to be unique.
    """

    # Python simulation of 2-node circuits
    fluxonium_N = 6
    c_specs = {'dimension': fluxonium_N, 'low': 0., 'high': 100, 'keep_prob': 1.0, 'keep_num': fluxonium_N}
    j_specs = {'dimension': fluxonium_N, 'low': 0., 'high': 200, 'keep_num': fluxonium_N}
    circuit_params = {'c_specs': c_specs, 'j_specs': j_specs, 'l_specs': None, 'phiOffs_specs': None}
    phiExt = np.linspace(0, 1, 41, endpoint=True)
    general_params = {'solver': '2-node', 'phiExt': phiExt}

    # Loss function settings
    with open('target_fluxqubit.p', 'rb') as content:
        target_info = pickle.load(content)
        phi = target_info['phiExt']
        # print(len(target_info['spectrum'][0]))
        target_info['num_states'] = 6  # FIXME Delete this once you figure out what num_states does.
        print(target_info)
    ts_options = {'target_spectrum': target_info['spectrum'], 'include_symmetry': True}

    # Initialize circuit searcher
    circuit_searcher = CircuitSearcher(circuit_params, general_params, database_path='Experiments')

    # Monte Carlo (random) optimization
    mc_options = {'max_iters': 6, 'max_concurrent': 2, 'batch_size': 10}
    computing_task_0 = circuit_searcher.add_task(
        name='random_search',
        designer='random', designer_options=mc_options,
        merit='TargetSpectrum', merit_options=ts_options)

    # Filtering for best circuits
    filtering_task_0 = circuit_searcher.add_task(name='filtering', designer='filter_db',
                                                 designer_options={'num_circuits': 2})

    # # L-BFGS-B optimization
    # bfgs_options = {'max_iters': 2, 'max_concurrent': 2}
    # ts_options = {'target_spectrum': target_info['spectrum'], 'include_symmetry': True}
    # computing_task_2 = circuit_searcher.add_task(
    #     name='lbfgs',
    #     designer='scipy', designer_options=bfgs_options,
    #     merit='TargetSpectrum', merit_options=ts_options, use_library=True)

    # Swarm Optimization
    swarm_options = {'max_iters': 2, 'max_concurrent': 2, 'n_particles': 2}
    # Loss function settings
    dw_options = {'max_peak': 1.5, 'max_split': 10, 'norm_p': 4, 'flux_sens': True, 'max_merit': 100}
    computing_task_2 = circuit_searcher.add_task(
        name='swarm_search',
        designer='particle_swarms', designer_options=swarm_options,
        merit='DoubleWell', merit_options=dw_options, use_library=True)

    tic_glob = time.time()
    # circuit_searcher.execute()

    # Initialize CircuitSearcher object for database readout
    circuit_reader = CircuitSearcher(database_path='Experiments')

    # Computing tasks (still no idea what this is)
    computing_tasks = circuit_reader.query(kind='list_computing_tasks')

    # All circuits sampled.
    sampled_circuits = circuit_reader.query(kind='get_circuits_from_task', task=computing_tasks[1])

    # print("Sampled Circuits")
    # for circuit in sampled_circuits:
    #     print(circuit)

    # Optimized trajectories using LBFGS.
    lbfgs_trajectories = circuit_reader.query(kind='get_trajectories', task=computing_tasks[2])

    traj_id = []

    print(f"\nFound {len(lbfgs_trajectories)} possible configurations:")
    for idd in lbfgs_trajectories:
        traj_id.append(idd)
        print(idd)
    print("")

    eigen_spec = lbfgs_trajectories[str(traj_id[0])][0]['merit']['measurements']['eigen_spectrum']

    size = [6.4 * 2, 4.8 * 2]

    plt.figure(figsize=size)
    # print(eigen_spec)
    plt.plot(eigen_spec)
    plt.xlabel("No. Arrays")
    plt.ylabel("Energy (GHz)")
    plt.title('L-BFGS Trajectories Eigen-Spectrum')
    plt.ylim(0, 1600)
    plt.savefig('EigenSpec1')

    # plt.figure(figsize=size)
    # plt.plot(sampled_circuits[-1]['merit']['measurements']['eigen_spectrum'])
    # plt.xlabel("No. Arrays")
    # plt.ylabel("Energy (GHz)")
    # plt.title('Sample Circuit Eigen-Spectrum')
    # plt.ylim(0, 1600)
    # plt.savefig('EigenSpec3')

    # eigen_mean = []
    #
    # for element in eigen_spec:
    #     eigen_mean.append(np.mean(element))
    #
    # plt.figure(figsize=size)
    # plt.plot(eigen_mean)
    # plt.xlabel("No. Arrays")
    # plt.ylabel("Energy (GHz)")
    # plt.title('L-BFGS Trajectories Eigen-Spectrum Mean')
    # plt.ylim(0, 1600)
    # plt.savefig('EigenSpec2')

    # print(lbfgs_trajectories[str(traj_id[0])][0]['circuit'])

    for i in range(len(traj_id)):
        print(f"\nConfiguration {i + 1}:")
        print(lbfgs_trajectories[str(traj_id[i])][0]['circuit']['circuit_values'])

    # jj = lbfgs_trajectories[str(traj_id[0])][0]['circuit']['circuit_values']['junctions']
    # cap = lbfgs_trajectories[str(traj_id[0])][0]['circuit']['circuit_values']['capacities']
    # x = np.linspace(min(phi), max(phi), 3)
    # plt.figure(figsize=size)
    # plt.scatter(x, jj, label="Josephson Junctions", marker='2')
    # plt.scatter(x, cap, label="Capacitors", marker="+")
    # plt.legend()
    # plt.xlabel("Phi")
    # plt.ylabel("Array Value")
    # plt.savefig('L-BFGS Trajectories CircuitConfig1')

    plot_basic_spec(target_info['spectrum'], 'Target Spectrum', "EigenSpec4", phi)

    for spectrum in range(len(traj_id)):
        plot_basic_spec(
            lbfgs_trajectories[str(traj_id[spectrum])][0]['merit']['measurements']['eigen_spectrum'],
            f'L-BFGS Trajectories {spectrum + 1}',
            f"EigenSpec{4 + 1 + spectrum}",
            phi
        )

    plt.figure(figsize=size)
    plt.plot(target_info['spectrum'])
    plt.xlabel("No. Arrays")
    plt.ylabel("Energy (GHz)")
    plt.title('Target Spectrum')
    plt.ylim(0, 1600)
    plt.savefig('EigenSpec0')

    print('\n####################################'
          '\n \n TOTAL TIME: {} s \n \n'
          '####################################'.format(time.time() - tic_glob))
