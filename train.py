import multiprocessing

import models

def evaluate_model(model_fn, train_size):
    from helpers import prepate_dataset, compute_weithts, image_generator, train_val_test, make_model, train, pickel_logs, show_metrics
    train_data, validation_data, test_data = prepate_dataset(train_size)
    initial_bias, class_weight = compute_weithts(train_data)

    model, preprocessing_function = model_fn(initial_bias)
    _, datagen = image_generator(preprocessing_function)
    train_generator, val_generator, test_generator = train_val_test(train_data, validation_data, test_data, datagen, datagen)
    model_compiled = make_model(model)


    print(f'\n\n\n-------------------------EVALUATING model {model_compiled.name}, train_size {train_size}-------------------------\n')
    history, evaluate = train(model_compiled, train_generator, val_generator, test_generator, class_weight)


    log_entry = {
        'model': history.model.name,
        'metrics': history.model.metrics_names,
        'history': history.history,
        'evaluation': evaluate,
        'train_size': train_size

    }
    pickel_logs(log_entry)

    show_metrics(history, evaluate, train_size)



def main():
    train_sizes = [5000, 4000, 3000, 2500, 2000, 1500, 1000, 500, 250, 100, 50]
    train_sizes = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

    for train_size in train_sizes:
        for model_fn in models.models_fn:
            p = multiprocessing.Process(target=evaluate_model,args=(model_fn, train_size,))
            p.start()
            p.join()
         


if __name__ == '__main__':
    main()