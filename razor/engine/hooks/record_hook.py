from razor.registry import HOOKS
from mmengine.hooks import Hook
@HOOKS.register_module()
class RecordHook(Hook):
    def __init__(self,
                 **kwargs):
        super(RecordHook, self).__init__(**kwargs)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs= None) -> None:
        iter = runner.iter
        #runner.model.recorder.data_idx=0
        """
        if iter%5==0 and iter!=0:
            if runner.model.recorder.data_idx <=1000:
                runner.model.recorder.data_idx += 10
        """
        if iter%5==0 and iter!=0:
            if runner.model.distiller.loss_forward_mappings.loss_kl1.preds_T.data_idx <=1000:
                runner.model.distiller.loss_forward_mappings.loss_kl1.preds_T.data_idx += 10
                runner.model.distiller.loss_forward_mappings.loss_kl2.preds_T.data_idx += 10
