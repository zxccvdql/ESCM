################################################################################################
# Training Utility Function
################################################################################################

from torch.autograd import Variable
import torch
import torch.distributed as dist
import copy

def train(net, dataset, criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, num_gpu = 1, scaler = None, max_iters=20000, save_model=True, log_file=None, method="foo", dest="/data", local_rank = 0):
    iteration = 0
    epoch = 0
    flag = False
    best_test_loss = None
    lr = 0
    save_int = 5000 if dataset == "taskonomy" else 100
    test_int = 100 if dataset == "taskonomy" else 50
    
    while True:
        total_loss = 0
        if 'module'in list(net.state_dict().keys())[0]:
            train_loader.sampler.set_epoch(epoch)
        for i, gt_batch in enumerate(train_loader):
            net.train()
            gt_batch["img"] = Variable(gt_batch["img"]).cuda()
            if "seg" in gt_batch:
                gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
            if "depth" in gt_batch:
                gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                if dataset == "taskonomy":
                    if 'depth_mask' in gt_batch.keys():
                        gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                    else:
                        print("No Depth Mask Existing. Please check")
                        gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
            if "normal" in gt_batch:
                gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
            if "keypoint" in gt_batch:
                gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
            if "edge" in gt_batch:
                gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
                
            # get the preds
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = net(gt_batch["img"])
                    loss = criterion(preds, gt_batch)
            else:
                preds = net(gt_batch["img"])
                loss = criterion(preds, gt_batch)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(optimizer)
                # optimizer.step()
                # scaler.step(scheduler)
                # scheduler.step()
                scaler.update()
                # if local_rank == 0:
                scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

            if (iteration+1) % 10 == 0:
                if local_rank == 0:
                    print(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                        % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                    log_file.write(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                    log_file.write("\n")
            # Save the model
            # if save_model:
            #     if local_rank == 0:
            #         if iteration % save_int == 0:
            #             print("Save checkpoint.")
            #             torch.save(net.state_dict(), f"{dest}/{iteration}th_{network_name}.pth")
            iteration += 1
            
            if iteration % 100 == 0:
                if local_rank == 0:
                    print(scheduler.get_last_lr())
                    log_file.write(f'{scheduler.get_last_lr()}')
                    log_file.write("\n")
            if iteration > max_iters:
                flag = True
                break
            
            # Validate on test dataset
            if iteration % test_int == 0:
                net.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for i, gt_batch in enumerate(test_loader):
                        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
                        if "seg" in gt_batch:
                            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
                        if "depth" in gt_batch:
                            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                            if dataset == "taskonomy":
                                if 'depth_mask' in gt_batch.keys():
                                    gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                                else:
                                    print("No Depth Mask Existing. Please check")
                                    gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
                        if "normal" in gt_batch:
                            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
                        if "keypoint" in gt_batch:
                            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
                        if "edge" in gt_batch:
                            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()

                        preds = net(gt_batch["img"])
                        loss = criterion(preds, gt_batch)
                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    test_loss = torch.tensor(test_loss).to('cuda')
                    if 'module'in list(net.state_dict().keys())[0]:
                        dist.reduce(test_loss, 0, op=dist.ReduceOp.SUM)
                    test_loss /= int(num_gpu)
                    if local_rank == 0:
                        print(f"{method}: TEST LOSS on {epoch}th epoch:", float(test_loss))
                        log_file.write(f"{method}: TEST LOSS on {epoch}th epoch: {test_loss}")
                        log_file.write("\n")
                    if save_model:
                        if best_test_loss is None:
                            best_test_loss = test_loss
                            if local_rank == 0:
                                print('Save first best model')
                                torch.save(net.state_dict(), f"{dest}/best_{network_name}.pth")
                        elif test_loss < best_test_loss:
                            if local_rank == 0:
                                print('Save best model')
                                torch.save(net.state_dict(), f"{dest}/best_{network_name}.pth")
                            best_test_loss = test_loss
                        elif test_loss >= best_test_loss:
                            if local_rank == 0:
                                print('test_loss > best_test_loss. do not save best model')
        epoch += 1

        # End Training
        if flag:
            break
    return net