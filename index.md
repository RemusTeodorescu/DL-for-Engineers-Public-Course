---
layout: default
title: DL for Engineers - Public Course
---

<div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;background:#000"><iframe src="assets/cover.html" title="DL for ENG" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;border:0"></iframe></div>

# DL for Engineers - Public Course

**Deep Learning for Engineering** - MSc course, AAU Energy, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant Noman Khan.

This repository holds the student-facing material: the lecture slides as PDF,
the lecture recordings as MP4 where available, and the exercise sets as
Colab-ready notebooks. It is generated from the course's development repository
and refreshed after every change, so it is always current.

## How to work with the exercises

Open any notebook and click the **Open in Colab** badge at its top. The first
code cell fetches the set's library files from this repository, so nothing needs
uploading. Work through the notebooks of a set in numerical order; each set ends
with a short report notebook. Running locally works just as well: clone the
repository and open the notebooks in Jupyter with the set folder as the working
directory.

## Lectures

Each lecture folder holds the slides (PDF) and, once recorded, the lecture video (MP4).
GitHub shows videos in the browser; use *Download raw file* on a video's page to keep a copy.

- `lectures/L00-course-intro/` - [slides](lectures/L00-course-intro/L0_Welcome_How_This_Course_Works.pdf)
- `lectures/L01.1-python-that-runs/` - [slides](lectures/L01.1-python-that-runs/L1.1_Python_That_Runs.pdf), [recording](lectures/L01.1-python-that-runs/L1.1_Python_That_Runs.mp4)
- `lectures/L01.2-numpy-and-arrays/` - [slides](lectures/L01.2-numpy-and-arrays/L1.2_NumPy_and_the_Shape_of_Everything.pdf), [recording](lectures/L01.2-numpy-and-arrays/L1.2_NumPy_and_the_Shape_of_Everything.mp4)
- `lectures/L02.1-scientific-python/` - [slides](lectures/L02.1-scientific-python/L2.1_Scientific_Python.pdf), [slides (v2, revised)](lectures/L02.1-scientific-python/L2.1_Scientific_Python_v2.pdf), [recording](lectures/L02.1-scientific-python/L2.1_Scientific_Python.mp4)
- `lectures/L02.2-pytorch-and-autograd/` - [slides](lectures/L02.2-pytorch-and-autograd/L2.2_PyTorch_as_an_Array_Library.pdf), [slides (v2, revised)](lectures/L02.2-pytorch-and-autograd/L2.2_PyTorch_as_an_Array_Library_v2.pdf), [recording](lectures/L02.2-pytorch-and-autograd/L2.2_PyTorch_as_an_Array_Library.mp4)
- `lectures/L03.1-what-is-ai/` - [slides](lectures/L03.1-what-is-ai/L3.1_What_is_AI.pdf), [recording](lectures/L03.1-what-is-ai/L3.1_What_is_AI.mp4)
- `lectures/L03.2-ai-application-examples/` - [slides](lectures/L03.2-ai-application-examples/L3.2_AI_Application_Examples.pdf), [recording](lectures/L03.2-ai-application-examples/L3.2_AI_Application_Examples.mp4)
- `lectures/L04.1-perceptron-to-neural-network/` - [slides](lectures/L04.1-perceptron-to-neural-network/L4.1_From_Perceptron_to_Neural_Network.pdf), [recording](lectures/L04.1-perceptron-to-neural-network/L4.1_From_Perceptron_to_Neural_Network.mp4)
- `lectures/L04.2-deep-neural-networks/` - [slides](lectures/L04.2-deep-neural-networks/L4.2_Deep_Neural_Networks.pdf), [recording](lectures/L04.2-deep-neural-networks/L4.2_Deep_Neural_Networks.mp4)
- `lectures/L05.1-convolutional-and-graph-networks/` - [slides](lectures/L05.1-convolutional-and-graph-networks/L5.1_Convolutional_and_Graph_Networks.pdf)
- `lectures/L05.2-sequences-and-attention/` - [slides](lectures/L05.2-sequences-and-attention/L5.2_Sequences_and_Attention.pdf)
- `lectures/L06.1-loss-functions-and-gradients/` - [slides](lectures/L06.1-loss-functions-and-gradients/L6.1_Loss_Functions_and_Gradients.pdf)
- `lectures/L06.2-post-training/` - [slides](lectures/L06.2-post-training/L6.2_Post_Training_and_Reinforcement_Learning.pdf)
- `lectures/L07.1-fundamentals-of-pinns/` - [slides](lectures/L07.1-fundamentals-of-pinns/L7.1_Fundamentals_of_PINNs.pdf)
- `lectures/L07.2-fundamental-pdes/` - [slides](lectures/L07.2-fundamental-pdes/L7.2_Fundamental_PDEs.pdf)
- `lectures/L08.1-stationary-heat/` - [slides](lectures/L08.1-stationary-heat/L8.1_Stationary_Heat.pdf)
- `lectures/L08.2-dynamic-heat-transfer/` - [slides](lectures/L08.2-dynamic-heat-transfer/L8.2_Dynamic_Heat_Transfer.pdf)
- `lectures/L09.1-laminar-flow/` - [slides](lectures/L09.1-laminar-flow/L9.1_Laminar_Flow.pdf)
- `lectures/L09.2-turbulent-flow/` - [slides](lectures/L09.2-turbulent-flow/L9.2_Turbulent_Flow.pdf)
- `lectures/L10.1-ionic-diffusion/` - [slides](lectures/L10.1-ionic-diffusion/L10.1_Ionic_Diffusion_Charge_Conservation.pdf)
- `lectures/L10.2-solid-oxide-cells/` - [slides](lectures/L10.2-solid-oxide-cells/L10.2_Solid_Oxide_Cells_Optimisation.pdf)
- `lectures/L11.1-vision-based-navigation/` - [slides](lectures/L11.1-vision-based-navigation/L11.1_Vision_Based_Navigation.pdf)
- `lectures/L11.2-dynamics-energy/` - [slides](lectures/L11.2-dynamics-energy/L11.2_Dynamics_Energy_and_Efficient_Driving.pdf)
- `lectures/L12.1-power-grid-stability-estimation/` - [slides](lectures/L12.1-power-grid-stability-estimation/L12.1_Power_Grid_Stability_Estimation.pdf)
- `lectures/L12.2-power-grid-stability-prediction/` - [slides](lectures/L12.2-power-grid-stability-prediction/L12.2_Power_Grid_Stability_Prediction.pdf)
- `lectures/_template/` - [slides](lectures/_template/LXX_Template.pdf)

## Exercises

Each set is a folder; open a notebook in Colab with its link, or browse the set on GitHub
to see the library files and the set's README.

- `exercises/Ex01-python-fundamentals/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex01-python-fundamentals)
  - Ex01_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_00_environment_check.ipynb)
  - Ex01_01_python_basics - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_01_python_basics.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_01_python_basics_light.ipynb)
  - Ex01_02_numpy_and_broadcasting - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_02_numpy_and_broadcasting.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_02_numpy_and_broadcasting_light.ipynb)
  - Ex01_03_plotting - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_03_plotting.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex01-python-fundamentals/Ex01_03_plotting_light.ipynb)
- `exercises/Ex02-pytorch-and-autograd/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex02-pytorch-and-autograd)
  - Ex02_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_00_environment_check.ipynb)
  - Ex02_01_tensors - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_01_tensors.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_01_tensors_light.ipynb)
  - Ex02_02_autograd_by_hand - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_02_autograd_by_hand.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_02_autograd_by_hand_light.ipynb)
  - Ex02_03_training_loop - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_03_training_loop.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex02-pytorch-and-autograd/Ex02_03_training_loop_light.ipynb)
- `exercises/Ex03-first-model/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex03-first-model)
  - Ex03_vibrating_mass - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex03-first-model/Ex03_vibrating_mass.ipynb)
- `exercises/Ex04-perceptron-to-mlp/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex04-perceptron-to-mlp)
  - Ex04_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_00_environment_check.ipynb)
  - Ex04_01_perceptron_and_xor - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_01_perceptron_and_xor.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_01_perceptron_and_xor_light.ipynb)
  - Ex04_02_pytorch_mlp - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_02_pytorch_mlp.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_02_pytorch_mlp_light.ipynb)
  - Ex04_03_counting_kinks - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_03_counting_kinks.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_03_counting_kinks_light.ipynb)
  - Ex04_04_depth_vs_width - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_04_depth_vs_width.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_04_depth_vs_width_light.ipynb)
  - Ex04_05_overfit_then_regularise - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_05_overfit_then_regularise.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_05_overfit_then_regularise_light.ipynb)
  - Ex04_06_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex04-perceptron-to-mlp/Ex04_06_report.ipynb)
- `exercises/Ex05-cnn-and-gnn/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex05-cnn-and-gnn)
  - Ex05_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_00_environment_check.ipynb)
  - Ex05_01_cnn_image_classification - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_01_cnn_image_classification.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_01_cnn_image_classification_light.ipynb)
  - Ex05_02_graph_basics - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_02_graph_basics.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_02_graph_basics_light.ipynb)
  - Ex05_03_gnn_six_bus_network - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_03_gnn_six_bus_network.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_03_gnn_six_bus_network_light.ipynb)
  - Ex05_04_sequence_model - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_04_sequence_model.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_04_sequence_model_light.ipynb)
  - Ex05_05_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex05-cnn-and-gnn/Ex05_05_report.ipynb)
- `exercises/Ex06-training-lab/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex06-training-lab)
  - Ex06_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_00_environment_check.ipynb)
  - Ex06_01_loss_functions - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_01_loss_functions.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_01_loss_functions_light.ipynb)
  - Ex06_02_optimiser_comparison - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_02_optimiser_comparison.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_02_optimiser_comparison_light.ipynb)
  - Ex06_03_transfer_and_fine_tuning - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_03_transfer_and_fine_tuning.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_03_transfer_and_fine_tuning_light.ipynb)
  - Ex06_04_quantisation - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_04_quantisation.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_04_quantisation_light.ipynb)
  - Ex06_05_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex06-training-lab/Ex06_05_report.ipynb)
- `exercises/Ex07.1-poisson/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex07.1-poisson)
  - Ex07.1_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_00_environment_check.ipynb)
  - Ex07.1_01_slot_soft_bc - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_01_slot_soft_bc.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_01_slot_soft_bc_light.ipynb)
  - Ex07.1_02_slot_hard_bc - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_02_slot_hard_bc.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_02_slot_hard_bc_light.ipynb)
  - Ex07.1_03_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_03_compare_and_report.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.1-poisson/Ex07.1_03_compare_and_report_light.ipynb)
- `exercises/Ex07.2-heat-and-wave/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex07.2-heat-and-wave)
  - Ex07.2_00_environment_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_00_environment_check.ipynb)
  - Ex07.2_01_die_soft_ic - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_01_die_soft_ic.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_01_die_soft_ic_light.ipynb)
  - Ex07.2_02_die_hard_ic - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_02_die_hard_ic.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_02_die_hard_ic_light.ipynb)
  - Ex07.2_03_panel_both_ics - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_03_panel_both_ics.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_03_panel_both_ics_light.ipynb)
  - Ex07.2_04_missing_condition - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_04_missing_condition.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_04_missing_condition_light.ipynb)
  - Ex07.2_05_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex07.2-heat-and-wave/Ex07.2_05_compare_and_report.ipynb)
- `exercises/Ex08.1-stationary-heat/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex08.1-stationary-heat)
  - Ex08.1_00_geometry_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_00_geometry_check.ipynb)
  - Ex08.1_01_manufactured - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_01_manufactured.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_01_manufactured_light.ipynb)
  - Ex08.1_02_plate_with_hole - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_02_plate_with_hole.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_02_plate_with_hole_light.ipynb)
  - Ex08.1_03_weights_and_flux - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_03_weights_and_flux.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_03_weights_and_flux_light.ipynb)
  - Ex08.1_04_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.1-stationary-heat/Ex08.1_04_compare_and_report.ipynb)
- `exercises/Ex08.2-transient-heat/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex08.2-transient-heat)
  - Ex08.2_00_transient_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_00_transient_check.ipynb)
  - Ex08.2_01_soft_ic - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_01_soft_ic.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_01_soft_ic_light.ipynb)
  - Ex08.2_02_hard_ic - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_02_hard_ic.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_02_hard_ic_light.ipynb)
  - Ex08.2_03_plate_transient - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_03_plate_transient.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_03_plate_transient_light.ipynb)
  - Ex08.2_04_inverse_alpha - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_04_inverse_alpha.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_04_inverse_alpha_light.ipynb)
  - Ex08.2_05_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex08.2-transient-heat/Ex08.2_05_compare_and_report.ipynb)
- `exercises/Ex09.1-burgers-2d/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex09.1-burgers-2d)
  - Ex09.1_00_setup_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_00_setup_check.ipynb)
  - Ex09.1_01_burgers2d - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_01_burgers2d.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_01_burgers2d_light.ipynb)
  - Ex09.1_02_control_panel - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_02_control_panel.ipynb)
  - Ex09.1_03_reynolds_sweep - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_03_reynolds_sweep.ipynb)
  - Ex09.1_04_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.1-burgers-2d/Ex09.1_04_report.ipynb)
- `exercises/Ex09.2-channel-flow/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex09.2-channel-flow)
  - Ex09.2_00_geometry_lab - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_00_geometry_lab.ipynb)
  - Ex09.2_01_channel_flow - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_01_channel_flow.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_01_channel_flow_light.ipynb)
  - Ex09.2_02_control_panel - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_02_control_panel.ipynb)
  - Ex09.2_03_shape_comparison - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_03_shape_comparison.ipynb)
  - Ex09.2_04_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex09.2-channel-flow/Ex09.2_04_report.ipynb)
- `exercises/Ex10.1-spm-and-thermal/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex10.1-spm-and-thermal)
  - Ex10.1_00_reference - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_00_reference.ipynb)
  - Ex10.1_01_spm - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_01_spm.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_01_spm_light.ipynb)
  - Ex10.1_02_spme - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_02_spme.ipynb)
  - Ex10.1_03_thermal - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_03_thermal.ipynb)
  - Ex10.1_04_control_panel - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_04_control_panel.ipynb)
  - Ex10.1_05_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.1-spm-and-thermal/Ex10.1_05_report.ipynb)
- `exercises/Ex10.2-solid-oxide-cell/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex10.2-solid-oxide-cell)
  - Ex10.2_00_cell_lab - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_00_cell_lab.ipynb)
  - Ex10.2_01_button_cell - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_01_button_cell.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_01_button_cell_light.ipynb)
  - Ex10.2_02_channel - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_02_channel.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_02_channel_light.ipynb)
  - Ex10.2_03_degradation - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_03_degradation.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_03_degradation_light.ipynb)
  - Ex10.2_04_optimisation - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_04_optimisation.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_04_optimisation_light.ipynb)
  - Ex10.2_05_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex10.2-solid-oxide-cell/Ex10.2_05_report.ipynb)
- `exercises/Ex11.1-camera-navigation/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex11.1-camera-navigation)
  - Ex11.1_10_camera_navigation - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex11.1-camera-navigation/Ex11.1_10_camera_navigation.ipynb)
- `exercises/Ex11.2-energy-optimization/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex11.2-energy-optimization)
  - Ex11.2_10_energy_optimization - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex11.2-energy-optimization/Ex11.2_10_energy_optimization.ipynb)
  - Ex11.2_20_energy_optimization_pinn - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex11.2-energy-optimization/Ex11.2_20_energy_optimization_pinn.ipynb)
- `exercises/Ex12.1-power-grid-stability-estimation/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex12.1-power-grid-stability-estimation)
  - Ex12.1_00_system_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_00_system_check.ipynb)
  - Ex12.1_01_wls_baseline - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_01_wls_baseline.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_01_wls_baseline_light.ipynb)
  - Ex12.1_02_algebraic_pinn - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_02_algebraic_pinn.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_02_algebraic_pinn_light.ipynb)
  - Ex12.1_03_dynamic_pinn - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_03_dynamic_pinn.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_03_dynamic_pinn_light.ipynb)
  - Ex12.1_04_inertia - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_04_inertia.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_04_inertia_light.ipynb)
  - Ex12.1_05_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.1-power-grid-stability-estimation/Ex12.1_05_compare_and_report.ipynb)
- `exercises/Ex12.2-power-grid-stability-prediction/` - [browse set](https://github.com/RemusTeodorescu/DL-for-Engineers-Public-Course/tree/main/exercises/Ex12.2-power-grid-stability-prediction)
  - Ex12.2_00_system_check - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_00_system_check.ipynb)
  - Ex12.2_01_contingency_set - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_01_contingency_set.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_01_contingency_set_light.ipynb)
  - Ex12.2_02_dense_baseline - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_02_dense_baseline.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_02_dense_baseline_light.ipynb)
  - Ex12.2_03_graph_network - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_03_graph_network.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_03_graph_network_light.ipynb)
  - Ex12.2_04_screening - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_04_screening.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_04_screening_light.ipynb)
  - Ex12.2_05_compare_and_report - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_05_compare_and_report.ipynb), [light](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_05_compare_and_report_light.ipynb)
  - Ex12.2_06_latency - [Colab](https://colab.research.google.com/github/RemusTeodorescu/DL-for-Engineers-Public-Course/blob/main/exercises/Ex12.2-power-grid-stability-prediction/Ex12.2_06_latency.ipynb)

## Licence

(c) 2026 Remus Teodorescu, Aalborg University. A licence will be added shortly;
until then the material is shared for the course's students.

Synced from the development repository; 28 lecture PDFs, 8 recordings and 18 exercise sets.
