import torch 
import numpy as np
import torch.distributed as dist

ood_k = 5
def compute_mean(features, task_id, device, args):
    # save statistics in mixture path
    # cluster features
    from sklearn.cluster import KMeans
    n_clusters = args.n_centroids
    features = features.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    cluster_lables = kmeans.labels_
    cluster_means = []
    for i in range(n_clusters):
        cluster_data = features[cluster_lables == i]
        cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
        cluster_means.append(cluster_mean)
    cluster_means = torch.stack(cluster_means, dim=0)
    task_feature_clusters[task_id] = cluster_means

def distance_metric(feat_1, feat_2, majority=False):
    # normalize
    feat_1 = torch.nn.functional.normalize(feat_1, p=2, dim=1).float()
    feat_2 = torch.nn.functional.normalize(feat_2, p=2, dim=1).float()
    norms = torch.cdist(feat_1, feat_2, p=2)
    if majority:
        return torch.min(norms, dim=1)[0].unsqueeze(-1)
    else:
        return torch.min(norms)

def group_task_features(task_id, device, args, features=None):
    distances = []
    majority = True
    if features is None:
        majority = False
        features = task_feature_clusters[task_id]
    for i in range(task_id):
        distance = distance_metric(features, task_feature_clusters[i], majority)
        distances.append(distance)
    if majority:
        distances = torch.cat(distances, dim=1)
        min_task_distance, min_task_id = torch.min(distances, dim=1)[0], torch.min(distances, dim=1)[1]
        min_index = torch.mode(min_task_id)[0]
        specific_task_distances = distances[:, min_index]
        specific_task_distances = torch.sort(specific_task_distances, descending=False)[0]
        min_distance = specific_task_distances[ood_k-1]
        print(min_index.item())
        print(specific_task_distances)
    if not majority and min(distances) <= args.cluster_threshold:
        min_index = [idx for idx, val in enumerate(distances) if val == min(distances)]
        task_domain_map[task_id] = task_domain_map[min_index]
    elif majority and min_distance <= args.cluster_threshold:
        task_domain_map[task_id] = task_domain_map[min_index.item()]
    else: 
        global domain_num
        domain_num += 1
        task_domain_map[task_id] = domain_num
 

@torch.no_grad()
def ood_detection(data_loader, vanilla_model, device, args, target_task_map):
    global task_domain_map
    global task_feature_clusters
    global domain_num
    domain_num = 0 
    task_domain_map = {}
    task_domain_map[0] = 0
    task_feature_clusters = {}
    for task_id in range(args.num_tasks):
        features_per_task = []
        for i, (inputs, targets) in enumerate(data_loader[task_id]['train']):
            inputs = inputs.to(device, non_blocking=True)
            features = vanilla_model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_task.append(features)
        features_per_task = torch.cat(features_per_task, dim=0)
        features_per_task_list = [torch.zeros_like(features_per_task, device=device) for _ in range(args.world_size)]

        dist.barrier()
        dist.all_gather(features_per_task_list, features_per_task)
        features_per_task_list = torch.cat(features_per_task_list, dim=0)
        compute_mean(features_per_task_list, task_id, device, args)
        if task_id > 0:
            group_task_features(task_id, device, args, features_per_task_list)

    # convert target_task_map and task_domain_map to target_domain_map
    target_domain_map = {}
    for target, task in target_task_map.items():
        target_domain_map[target] = task_domain_map[task]
    shuffle_task_domain_map = {}
    for key, val in task_domain_map.items():
        shuffle_task_domain_map[args.tasks_order[key]] = val
    print(shuffle_task_domain_map)
    return target_domain_map, domain_num + 1, task_feature_clusters

