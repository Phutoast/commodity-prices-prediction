from models.ind_multi_model import IndependentMultiModel
from models.GP_multi_out import GPMultiTaskMultiOut
from models.GP_multi_index import GPMultiTaskIndex

# For multi-task Learner Only
multi_task_algo = {
    "IndependentMultiModel": IndependentMultiModel,
    "GPMultiTaskMultiOut": GPMultiTaskMultiOut,
    "GPMultiTaskIndex": GPMultiTaskIndex
}