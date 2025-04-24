from demo_score.demo_score.filter_sweep import analyzer_classifier_sweep_and_filter
from .step3_classifier_sweep import eps_dict

if __name__ == '__main__':

    analyzer_classifier_sweep_and_filter(expt_dir='data/example',
                                        run_name='ex1_seed10000',
                                        eps_dict=eps_dict,
                                        dataset_type='lerobot',
                                        model_arch='small_stepwise')