import os
import subprocess
from glob import glob

import pytest
import torch


os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')


@pytest.fixture
def train_args():
    return {
        'trainer.devices': torch.cuda.device_count(),
        'trainer.num_nodes': 1,
        'trainer.max_steps': 20,
        'trainer.val_check_interval': 10,
        'trainer.limit_val_batches': 2,
        'exp_manager.create_wandb_logger': False,
        'exp_manager.create_tensorboard_logger': False,
        'model.data.dataset.train': 'x000',
        'model.data.dataset.val': 'x000',
        'model.data.dataset.test': 'x000',
        'model.micro_batch_size': 2,
    }


# TODO: can we assume that we always run these tests from main bionemo dir?
DIRS_TO_TEST = [
    'examples/',
    'examples/molecule/megamolbart/',
    'examples/protein/downstream/',
    'examples/protein/esm1nv/',
    'examples/protein/prott5nv/',
    'examples/protein/openfold/',
]

TRAIN_SCRIPTS = []
for subdir in DIRS_TO_TEST:
    TRAIN_SCRIPTS += list(glob(os.path.join(subdir, '*train*.py')))
    TRAIN_SCRIPTS += [f for f in glob(os.path.join(subdir, 'downstream*.py')) if not f.endswith('test.py')]


INFERENCE_CONFIGS = []
for subdir in DIRS_TO_TEST:
    INFERENCE_CONFIGS += list(glob(os.path.join(subdir, 'conf', 'infer*yaml')))


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    TEST_DATA_DIR = '/workspace/bionemo/examples/tests/test_data'
    MAIN = f' ++model.data.dataset_path={TEST_DATA_DIR}/%s'
    DOWNSTREAM = f' ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s'

    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    assert root == 'examples' and model in (
        'megamolbart',
        'esm1nv',
        'prott5nv',
        'downstream',
        'openfold',
    ), 'update this function, patterns might be wrong'

    task = {
        'molecule': 'physchem/SAMPL',
        'protein': 'downstream',
    }

    if conf == ['conf']:
        if model in ('megamolbart', 'openfold'):
            return ''
        else:
            return MAIN % f'{domain}/{task[domain]}/test/x000'

    if 'retro' in script:
        return MAIN % 'reaction'
    elif model == 'openfold':
        return MAIN % 'openfold_data'
    elif 'downstream' in script:
        return MAIN % f'{domain}/{task[domain]}'
    else:
        return (MAIN + DOWNSTREAM) % (domain, f'{domain}/{task[domain]}')


def get_train_args_overrides(script_or_cfg_path, train_args):
    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    if model == "openfold":
        # FIXME: provide even smaller data sample or do not generate MSA features
        pytest.skip(reason="CI infrastructure is too limiting")
        train_args['model.micro_batch_size'] = 1
        train_args['model.train_ds.num_workers'] = 1
        train_args['model.train_sequence_crop_size'] = 32
        # do not use kalign as it requires third-party-download and it not essential for testing
        train_args['model.data.realign_when_required'] = False
    return train_args


@pytest.mark.needs_gpu
@pytest.mark.parametrize('script_path', TRAIN_SCRIPTS)
def test_train_scripts(script_path, train_args, tmp_path):
    data_str = get_data_overrides(script_path)
    train_args = get_train_args_overrides(script_path, train_args)
    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    assert process_handle.returncode == 0


def get_infer_args_overrides(config_path):
    if 'openfold' in config_path:
        return {
            # cropped 7YVT_B  # cropped 7ZHL
            # predicting on longer sequences will result in CUDA OOM.
            # TODO: if preparing MSA is to be tested, the model has to be further scaled down
            'sequences': r"\['GASTATVGRWMGPAEYQQMLDTGTVVQSSTGTTHVAYPAD','MTDSIKTLSAHRSFGGVQHFHEHASREIGLPMRFAAYLPP'\]"
        }
    return {}


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('config_path', INFERENCE_CONFIGS)
def test_infer_script(config_path, tmp_path):
    config_dir, config_name = os.path.split(config_path)
    script_path = os.path.join(os.path.dirname(config_dir), 'infer.py')
    infer_args = get_infer_args_overrides(config_path)
    if not os.path.exists(script_path):
        script_path = 'examples/infer.py'
    cmd = (
        f'python {script_path} --config-dir {config_dir} --config-name {config_name} ++exp_manager.exp_dir={tmp_path} '
        + ' '.join(f'++{k}={v}' for k, v in infer_args.items())
    )

    # FIXME: WARs for unavailable checkpoints
    if 'retro' in config_path:
        cmd += ' model.downstream_task.restore_from_path=/model/molecule/megamolbart/megamolbart.nemo'
    cmd += get_data_overrides(config_path)
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    assert process_handle.returncode == 0
