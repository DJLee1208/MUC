from yacs.config import CfgNode as CN

_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node. per node -> 서버 여러개 쓸 때 
_C.NUM_GPUS = 1
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = './results/'
_C.LOG_TIME = True

_C.DATA_LOADER = CN()
# the number of data loading workers per gpu
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True
_C.DATA_LOADER.PREFETCH_FACTOR = 2
_C.DATA_LOADER.PERSISTENT_WORKERS = True


_C.DATA = CN()
_C.DATA.BASE_DIR = './data/'
_C.DATA.NAME = 'weather' #! 체크
_C.DATA.N_VAR = 24 #! 체크
_C.DATA.SEQ_LEN = 96 # encoder에 들어가는 window 길이 (look back window 보통 고정)
_C.DATA.LABEL_LEN = 48 # decoder embedding으로 들어가는 길이 (seq_len 의 절반이네 여기선) (itransformer에서는 필요없음)
_C.DATA.PRED_LEN = 96 # prediction 길이 (이걸 변화시킨다 주로)
_C.DATA.FEATURES = 'M' # prediction target이 multivariate이고 'S'는 input은 multivariate 예측 univariate
_C.DATA.TIMEENC = 0 # data에 time stamp가 어떻게 되었냐에 따라 0, 1
_C.DATA.FREQ = 'h' # h or t. temporal embedding. t는 minute embedding 포함. h는 hour, weekday, day, month embedding (iTransformer에서는 필요없음)
_C.DATA.SCALE = "standard" # 초기 전처리 normalization 방법 # standard, min-max
_C.DATA.TRAIN_RATIO = 0.7 # train, val, test 비율, 데이터는 train, val, test 순서대로 자름
_C.DATA.TEST_RATIO = 0.2 # train, val, test 비율
_C.DATA.DATE_IDX = 0 # raw data에서 date가 있지만 날려야 되니까 날리는 column index 
_C.DATA.TARGET_START_IDX = 0 # column 날린기준, prediction target이 시작하는 column index (예측하는 변수가 뒤에 몰려있어야 함)

_C.TRAIN = CN()
_C.TRAIN.ENABLE = True # main.py에서 training 할건지 여부
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.BATCH_SIZE = 64 #! 체크
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True  # 전체 dataset 길이가 batch_size로 나누어 떨어지지 않을 때 마지막 batch를 버릴지 여부
_C.TRAIN.CHECKPOINT_DIR = './checkpoints/' # directory to save checkpoints
_C.TRAIN.RESUME = '' # path to checkpoint to resume training
_C.TRAIN.CHECKPOINT_PERIOD = 200 # epoch period to save checkpoints
_C.TRAIN.EVAL_PERIOD = 1 # epoch period to evaluate on a validation set
_C.TRAIN.PRINT_FREQ  = 100 # iteration frequency to print progress meter
_C.TRAIN.BEST_METRIC_INITIAL = float("inf") # MSE 나 MAE로 재는데 best model tracking 하기 위한거라서 초기값은 무한대로
_C.TRAIN.BEST_LOWER = True # best metric이 낮을수록 좋은지 높을수록 좋은지
############################    MUC    ############################
_C.TRAIN.MACs_weight = 3e-9  #! 체크 이렇게 줄 수도 있고 loss에 들어가게 아니면 MACs contraint 걸어도 될듯
_C.TRAIN.LASSO_weight = 1e-2 # 좀 줄였어도 될듯 #! 체크
############################    MUC    ############################

_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = 64 #! 체크
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.VIS = False

_C.TEST = CN()
_C.TEST.ENABLE = True # main.py에서 test 할건지 여부
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 1 #!체크
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False
_C.TEST.VIS_ERROR = True # Error 보여줄건지
_C.TEST.VIS_DATA = True # TOP, WORST data 보여줄건지
_C.TEST.VIS_DATA_NUM = 5 # TOP, WORST 몇개 보여줄건지
_C.TEST.PREDICTION_ERROR_DIR = "" # 불러와서 쓰고 싶으면, 평상시에는 ""로 놓고
_C.TEST.PREDICTION_ERROR_TYPE = "MAE" # MAE, MSE

_C.TEST.APPLY_MOVING_AVERAGE = False # True로 하면 moving average 적용, 시간에 따른 변화를 보기 위함 smoothing (TTA 관련 체크를 위해)
_C.TEST.MOVING_AVERAGE_WINDOW = 100 


_C.MODEL_NAME = 'iTransformer_MUC' #! 체크
_C.MODEL = CN()
############################    MUC    ############################
_C.MODEL.embed_depth = 3
############################    MUC    ############################
_C.MODEL.task_name = 'long_term_forecast'
_C.MODEL.seq_len = _C.DATA.SEQ_LEN 
_C.MODEL.label_len = _C.DATA.LABEL_LEN # iTransformer에서는 필요 없음
_C.MODEL.pred_len = _C.DATA.PRED_LEN 
_C.MODEL.e_layers = 3
_C.MODEL.d_layers = 1 # iTransformer에서는 필요 없음
_C.MODEL.factor = 3 # Prob Attention(probabilistic attention)에서 쓰는데 informer에서 씀
_C.MODEL.num_kernels = 6 # for Inception

_C.MODEL.enc_in = _C.DATA.N_VAR # classification에서만 쓰임
_C.MODEL.dec_in = _C.DATA.N_VAR # iTransformer에서는 필요 없음
_C.MODEL.c_out = _C.DATA.N_VAR # iTransformer에서는 필요 없음

_C.MODEL.d_model = 512 # embedding dimension
_C.MODEL.d_ff = 512 # 2048 # feedforward dimension d_model -> d_ff -> d_model

_C.MODEL.moving_avg = 25 # window size of moving average 라는데 autoformer에서 쓰는거 같다

_C.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_C.MODEL.dropout = 0.1 # 보통 0.1 많이 쓰는 것 같다
_C.MODEL.n_heads = 8
_C.MODEL.activation = 'gelu'
_C.MODEL.METRIC_NAMES = ('MAE',)
_C.MODEL.LOSS_NAMES = ('MSE',)

# positional embedding은 window안에서의 포지션
# temporal embedding은 시간에 대한 정보
_C.MODEL.embed = 'timeF' # iTransformer에서는 필요 없음
_C.MODEL.freq = 'h' # iTransformer에서는 필요 없음 


_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 40
_C.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SOLVER.BASE_LR = 0.0003 # warmup end learning rate
_C.SOLVER.WEIGHT_DECAY = 3.0e-05 #1e-4
_C.SOLVER.LR_POLICY = 'decay' # 없애면 base_lr로 돌아감
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = False # warmup 없는 cosine만 하려면 false, warmup 있는 cosine 하려면 true
_C.SOLVER.WARMUP_EPOCHS = 0.2 # linear warmup epoch
_C.SOLVER.WARMUP_START_LR = 0 # warmup start learning rate

_C.SOLVER.LR_DECAY_STEP = 1
_C.SOLVER.LR_DECAY_RATE = 0.9


# learning rate of last fc layer is scaled by fc_lr_ratio
# _C.SOLVER.FC_LR_RATIO = 10.0


_C.WANDB = CN()
_C.WANDB.ENABLE = True # wnadb on/off #! 체크
_C.WANDB.PROJECT = 'MUC'
_C.WANDB.NAME = 'iTransformer_MUC_Weather' #! 체크
_C.WANDB.JOB_TYPE = 'train' # train or eval
_C.WANDB.NOTES = '' # a description of this run
_C.WANDB.DIR = './'
_C.WANDB.VIS_TRAIN_SCORE = False
_C.WANDB.VIS_TEST_SCORE = False 
_C.WANDB.VIS_TEST_DATA = False # raw data 찍기
_C.WANDB.VIS_TRAIN_TEST_HISTOGRAM = False

def get_cfg_defaults():
    return _C.clone()