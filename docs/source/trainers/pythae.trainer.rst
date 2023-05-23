**********************************
Trainers
**********************************

.. toctree::
   :hidden:
   :maxdepth: 1

   pythae.trainers.base_trainer
   pythae.trainers.coupled_trainer
   pythae.trainers.adversarial_trainer
   pythae.trainers.coupled_optimizer_adversarial_trainer
   pythae.training_callbacks
    

.. automodule::
   pythae.trainers
    

.. autosummary::
   ~pythae.trainers.BaseTrainer
   ~pythae.trainers.CoupledOptimizerTrainer
   ~pythae.trainers.AdversarialTrainer
   ~pythae.trainers.CoupledOptimizerAdversarialTrainer
   :nosignatures:

----------------------------------
Training Callbacks
----------------------------------

.. automodule::
   pythae.trainers.training_callbacks

.. autosummary::
   ~pythae.trainers.training_callbacks.TrainingCallback
   ~pythae.trainers.training_callbacks.CallbackHandler
   ~pythae.trainers.training_callbacks.MetricConsolePrinterCallback
   ~pythae.trainers.training_callbacks.ProgressBarCallback
   ~pythae.trainers.training_callbacks.WandbCallback
   ~pythae.trainers.training_callbacks.MLFlowCallback
   ~pythae.trainers.training_callbacks.CometCallback
   :nosignatures: