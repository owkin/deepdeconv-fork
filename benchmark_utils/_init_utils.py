from scvi.model import SCVI, MixUpVI_v2
import logging

logger = logging.getLogger(__name__)

def transfer_weights_selective(base_model: SCVI, mixupvi_model: MixUpVI_v2) -> None:
    """
    Transfer weights from SCVI to MixUpVI_v2, handling the different module structures.
    
    Parameters
    ----------
    scvi_model : SCVI
        The source SCVI model
    mixupvi_model : MixUpVI_v2
        The target MixUpVI_v2 model
    """
    base_module = base_model.module
    mixupvae_module = mixupvi_model.module
    
    # Get state dicts
    base_state_dict = base_module.state_dict()
    mixupvae_state_dict = mixupvae_module.state_dict()

    # Create a new state dict for MixUpVAE_v2
    new_state_dict = {}
    
    for i, (name, param) in enumerate(base_state_dict.items()):
        if name in mixupvae_state_dict:
            if mixupvae_state_dict[name].shape == param.shape:
                new_state_dict[name] = param.clone()
            else:
                #TODO: Error handling
                #print(f"Shape mismatch for {name}, the {i}th parameter: VAE {param.shape} vs MixUpVAE_v2 {mixupvae_state_dict[name].shape}")
                logger.error(f"Shape mismatch for {name}, the {i}th parameter: Base Module {param.shape} vs MixUpVAE_v2 {mixupvae_state_dict[name].shape}")
                raise ValueError(f"Shape mismatch for {name}, the {i}th parameter: Base Module {param.shape} vs MixUpVAE_v2 {mixupvae_state_dict[name].shape}")
        else:
            logger.error(f"Parameter {name} not found in MixUpVAE_v2")
            raise ValueError(f"Parameter {name} not found in MixUpVAE_v2")
    
    # Load the transferred weights
    mixupvae_module.load_state_dict(new_state_dict, strict=False)
    logger.info("Weight transfer completed (non-strict mode)")