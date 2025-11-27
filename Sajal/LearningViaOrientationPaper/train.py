import numpy as np
import torch
import torch.optim as optim
from models import MODELS
import torch.utils.data as data
from torch.autograd import Variable
import util
from config import config
from loss import mIoULoss, CrossEntropyLoss2d
from dataset import DeepGlobeDataset

train_loss_file = open('train.txt', "w+")
val_loss_file = open('val.txt', "w+")
train_loss_angle_file = open("train_angle.txt", "w+")
val_loss_angle_file = open('val_angle.txt', "w+")


MODEL_CHOOSEN = "LinkNet34MTL" #"StackHourglassNetMTL"
multi_scale_pred = False

num_gpus = torch.cuda.device_count()

model = MODELS[MODEL_CHOOSEN](
    config["task1_classes"], config["task2_classes"]
)
model.cuda()

optimizer = optim.SGD(
    model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005
)

train_loader = data.DataLoader(
    DeepGlobeDataset(
        config["train_dataset"],
        seed=config["seed"],
        is_train=True,
        multi_scale_pred=multi_scale_pred,
    ),
    batch_size=config["train_batch_size"],
    num_workers=1,
    shuffle=True,
    pin_memory=False,
)
weights = torch.ones(config["task1_classes"]).cuda()
if config["task1_weight"] < 1:
    print("Roads are weighted.")
    weights[0] = 1 - config["task1_weight"]
    weights[1] = config["task1_weight"]

road_loss = mIoULoss(
    weight=weights, size_average=True, n_classes=config["task1_classes"]
).cuda()

weights_angles = torch.ones(config["task2_classes"]).cuda()
if config["task2_weight"] < 1:
    print("Road angles are weighted.")
    weights_angles[-1] = config["task2_weight"]

angle_loss = CrossEntropyLoss2d(
    weight=weights_angles, size_average=True, ignore_index=255, reduce=True
).cuda()

train_loss_iou = 0
train_loss_vec = 0

model.train()

optimizer.zero_grad()

hist = np.zeros((config["task1_classes"], config["task1_classes"]))
hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
crop_size = config["train_dataset"]["deepglobe"]["crop_size"]

for epoch in range(30):
  for i, data in enumerate(train_loader, 0):
          inputsBGR, labels, vecmap_angles = data
          inputsBGR = Variable(inputsBGR.float().cuda())
          outputs, pred_vecmaps = model(inputsBGR)
          print(outputs.shape, pred_vecmaps.shape)

          if multi_scale_pred:
              pass
          else:
              loss1 = road_loss(outputs, util.to_variable(labels[-1], requires_grad = False), False)
              loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles[-1], requires_grad = False))

          train_loss_iou += loss1.item()
          train_loss_vec += loss2.item()

          _, predicted = torch.max(outputs.data, 1)

          correctLabel = labels[-1].view(-1, crop_size, crop_size).long()

          hist += util.fast_hist(
              predicted.view(predicted.size(0), -1).cpu().numpy(),
              correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
              config["task1_classes"],
          )

          _, predicted_angle = torch.max(pred_vecmaps.data, 1)
          correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
          hist_angles += util.fast_hist(
              predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
              correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
              config["task2_classes"],
          )

          p_accu, miou, road_iou, fwacc = util.performMetrics(
              train_loss_file,
              val_loss_file,
              epoch,
              hist,
              train_loss_iou / (i + 1),
              train_loss_vec / (i + 1),
          )

          p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
              train_loss_angle_file, val_loss_angle_file, epoch, hist_angles
          )


          torch.autograd.backward([loss1, loss2])

          if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
              optimizer.step()
              optimizer.zero_grad()

          del (
              outputs,
              pred_vecmaps,
              predicted,
              correct_angles,
              correctLabel,
              inputsBGR,
              labels,
              vecmap_angles,
          )


  print(util.performMetrics(
          train_loss_file,
          val_loss_file,
          epoch,
          hist,
          train_loss_iou / len(train_loader),
          train_loss_vec / len(train_loader),
          write=True,
      ))

  util.performAngleMetrics(
      train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, write=True
  )


