
def score(self, candidate, split='train', write=False):

    label_probs, calibrated_label_probs, raw_acc_count, raw_cal_acc_count, answer_list, index_list, _ = run(
        mode=mode, batch_size=batch_size, num_shots=num_shots, chosen_task_name=chosen_task_name,
        num_samples=num_samples, seed=seed, override_prompts=True, function=custom_instruction_prompt, split=split,
        modified={'Definition': candidate}, task_labels=task_labels, if_calibrate=False)
    preds = get_prediction(label_probs, task_labels)
    raw_acc = balanced_accuracy_score(answer_list, preds)
    label_frequencies = [preds.count(l) / len(preds) for l in task_labels]
    if split == 'train':
        return np.round(100 * raw_acc, 2) + 10 * entropy(label_frequencies)
    elif split == 'test':
        if write:
            pname = args.meta_name
            pname = pname.split('.')[0] + "_predictions.json"
            pred_dump = {'predictions': preds, 'answers': answer_list, 'ids': index_list}
            ppath = os.path.join(args.meta_dir, pname)
            pfile = open(ppath, 'w+')
            json.dump(pred_dump, pfile)
        return np.round(100 * raw_acc_count / len(answer_list), 2)
    else:
        return