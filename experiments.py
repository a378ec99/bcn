import sys

import numpy as np



def test(run):
    seed = np.random.randint(0, 1e8)
    print seed
 
    if run == 'entry_sparsity-1_rank':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'known_correlations': 1.0,
                      'operator_name': 'entry',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 1,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 5), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(3000), np.log10(1.1e4), 5), dtype=int)))}
    
    if run == 'dense_sparsity-2_rank':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))} # WARNING too much is 5.0e6.
    
    if run == 'dense_sparsity-2_rank-2_known_correlations2':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'rank': 2,
                      'free_x': ('known_correlations', list(np.logspace(np.log10(0.8), np.log10(1), 10))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(10), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'dense_sparsity-2_rank-2_additive_noise_A_std':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',  # 'parallel'
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_y_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'rank': 2,
                      'known_correlations': 1.0,
                      'free_x': ('additive_noise_A_std', list(np.logspace(np.log10(1e-4), np.log10(1e2), 5))), #NOTE Mid range is 0.1
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(3000), np.log10(1.1e4), 5), dtype=int)))}
    
    if run == 'dense_rank-2_sparsity':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'additive_noise_y_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'rank': 2,
                      'known_correlations': 1.0,
                      'free_x': ('sparsity', list(np.asarray(np.logspace(np.log10(1), np.log10(30), 5), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.0e4), 5), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 5), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(10), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank-2_known_correlations2':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'rank': 2,
                      'free_x': ('known_correlations', list(np.logspace(np.log10(0.8), np.log10(1), 10))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(10), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank999':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank_replicate999':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank_noise-1':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 2,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 2,
                      'noise_amplitude': 1.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'custom_sparsity-2_rank-2_correlation_strength22':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'free_x': ('correlation_strength', list(np.linspace(0.2, 1.0, 10))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'dense_sparsity-2_rank-2_correlation_strength222':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'free_x': ('correlation_strength', list(np.linspace(0.0, 1.0, 10))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'dense_sparsity-2_rank-2_additive_noise_y_std222bc':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'free_x': ('additive_noise_y_std', list(np.logspace(np.log10(0.001), np.log10(0.01), 5))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 5), dtype=int)))}
                      
    if run == 'real_4_custom_ideal_rank':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'real_4_custom_ideal_rank_big':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (500, 550),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'real_4_custom_ideal_rank_huge':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 10,
                      'shape': (1000, 1100),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'custom',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
                                          
    if run == 'real_4_dense_sparsity-1_rankb':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 1,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 1,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 2), dtype=int))), # 10
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 2), dtype=int)))} # 10
    
    if run == 'real_4_dense_sparsity-2_rankb':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 1,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'real_4_dense_sparsity-10_rankb':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 1,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 10,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    if run == 'real_4_dense_sparsity-20_rankb':
        parameters = {'class': 'Experiment',
                      'name': 'test_' + run,
                      'mode': 'parallel',
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 1,
                      'shape': (100, 110),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 20,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'known_correlations': 1.0,
                      'rank': 2,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 10, 10), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(100), np.log10(1.1e4), 10), dtype=int)))}
    
    submit(parameters)


if __name__ == '__main__':
    arg = sys.argv[1]
    test(arg)
