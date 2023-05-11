# SPMKD
In this paper, we present "A Self-supervised Pressure Map Keypoint Detection Approach: Balancing Generalization and Computational Efficiency across Tasks and Datasets." Our proposed method, referred to as Self-supervised Pressure Map Keypoint Detection (SPMKD), is specifically designed to accommodate the unique image characteristics of pressure maps. The SPMKD approach comprises two main components: an Encoder-Fuser-Decoder (EFD) model, and a Classification-to-Regression Weight Transfer (CRWT) technique. When evaluated on the SLP and SMaL datasets, our model demonstrates superior computational efficiency and generalization performance compared to conventional approaches.

## Dataset download
- SLP dataset: https://github.com/ostadabbas/SLP-Dataset-and-Code
- SMaL dataset: https://ieee-dataport.org/documents/simultaneously-collected-multimodal-mannequin-lying-pose-smal

## Weight download
As the weights file exceeds the Github limit, we use anonymity to share the weights we have trained. All you need to do is download the `weights` folder and place it in the root of your project.

- google drive: https://drive.google.com/drive/folders/15OlgVcOVyzdQz_J-uP6oYbodGPUVO70y?usp=sharing

## The effect analysis of whether to apply the classifier or not
- ResNet `python main.py --phase test --detection_head ResNet`
- DenseNet `python main.py --phase test --detection_head DenseNet`
- SPMKD+ResNet `python main.py --phase test --detection_head ResNet --use_Encoder`
- SPMKD+DenseNet `python main.py --phase test --detection_head DenseNet --use_Encoder`

## On the SLP dataset, self-supervised keypoints vs manual annotation-based keypoints

- SPMKD+GCN+SLP `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GCN`
- SPMKD+GAT+SLP `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GAT`
- SPMKD+GraphSAGE+SLP `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GraphSAGE`


## On unfamiliar datasets, self-supervised keypoints vs manual annotation-based keypoints
- SPMKD+GCN+SMaL `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GCN --SMaL`
- SPMKD+GAT+SMaL `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GAT --SMaL`
- SPMKD+GraphSAGE+SMaL `python main.py --phase test --use_Keypoint --keypoint_type Ours --GNN_Network GraphSAGE --SMaL`