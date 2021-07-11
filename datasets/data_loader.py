import office31
import digit5
import ASD
import os


def data_loader(dataset_name, folder, batch_size, infinite_data_loader, train, num_workers):

    if dataset_name == 'office31':
        source_loader, n_class = office31.load_data(data_folder=folder, batch_size=batch_size,
                                                    infinite_data_loader=infinite_data_loader,
                                                    train=train, num_workers=num_workers)
        source_test_loader = None
    elif dataset_name == 'ASD':
        source_loader, n_class = ASD.load_data(data_folder=folder, batch_size=batch_size,
                                               infinite_data_loader=infinite_data_loader,
                                               train=train, num_workers=num_workers)
        source_test_loader = None
        
    elif dataset_name == 'digit5':
        source_loader, source_test_loader, n_class = digit5.load_data(domain_name=folder, batch_size=batch_size,
                                                                      infinite_data_loader=infinite_data_loader,
                                                                      num_workers=num_workers)

    else:
        source_loader, source_test_loader, n_class = None, None, None
        print('There is no dataset named ' + dataset_name)

    return source_loader, source_test_loader, n_class


def data_read(dataset_name, target_domain, batch_size, infinite_data_loader, train, num_workers):
    
    if dataset_name == 'office31':
        root = '../data/office31'
        
        source_domains = ['amazon', 'dslr', 'webcam']
        source_domains.remove(target_domain)
        
        target_folder = os.path.join(root, target_domain)
        target_loader, _, n_class = data_loader(dataset_name=dataset_name, folder=target_folder,
                                                batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                                train=train, num_workers=num_workers)
        
        source_folder_1 = os.path.join(root, source_domains[0])
        source_folder_2 = os.path.join(root, source_domains[1])

        source_loader_1, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_1,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)
        source_loader_2, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_2,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)

        return source_loader_1, source_loader_2, target_loader, n_class
    
    elif dataset_name == 'ASD':
        root = '../data/ASD'
        data = 'RSFC.mat'
        
        source_domains = ['LEUVEN', 'NYU', 'UCLA', 'UM', 'USM', 'YALE']
        source_domains.remove(target_domain)

        target_folder = os.path.join(root, target_domain, data)
        target_loader, _, n_class = data_loader(dataset_name=dataset_name, folder=target_folder,
                                                batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                                train=train, num_workers=num_workers)

        source_folder_1 = os.path.join(root, source_domains[0], data)
        source_folder_2 = os.path.join(root, source_domains[1], data)
        source_folder_3 = os.path.join(root, source_domains[2], data)
        source_folder_4 = os.path.join(root, source_domains[3], data)
        source_folder_5 = os.path.join(root, source_domains[4], data)

        source_loader_1, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_1,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)
        source_loader_2, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_2,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)
        source_loader_3, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_3,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)
        source_loader_4, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_4,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)
        source_loader_5, _, _ = data_loader(dataset_name=dataset_name, folder=source_folder_5,
                                            batch_size=batch_size, infinite_data_loader=infinite_data_loader,
                                            train=train, num_workers=num_workers)

        return source_loader_1, source_loader_2, source_loader_3, \
               source_loader_4, source_loader_5, target_loader, n_class

    elif dataset_name == 'digit5':
        source_domains = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
        source_domains.remove(target_domain)

        target_loader, target_test_loader, n_class = data_loader(dataset_name=dataset_name, folder=target_domain,
                                                                 batch_size=batch_size,
                                                                 infinite_data_loader=infinite_data_loader,
                                                                 train=None, num_workers=num_workers)

        source_folder_1 = source_domains[0]
        source_folder_2 = source_domains[1]
        source_folder_3 = source_domains[2]
        source_folder_4 = source_domains[3]

        source_loader_1, source_test_loader_1, _ = data_loader(dataset_name=dataset_name, folder=source_folder_1,
                                                               batch_size=batch_size,
                                                               infinite_data_loader=infinite_data_loader,
                                                               train=None, num_workers=num_workers)
        source_loader_2, source_test_loader_2, _ = data_loader(dataset_name=dataset_name, folder=source_folder_2,
                                                               batch_size=batch_size,
                                                               infinite_data_loader=infinite_data_loader,
                                                               train=None, num_workers=num_workers)
        source_loader_3, source_test_loader_3, _ = data_loader(dataset_name=dataset_name, folder=source_folder_3,
                                                               batch_size=batch_size,
                                                               infinite_data_loader=infinite_data_loader,
                                                               train=None, num_workers=num_workers)
        source_loader_4, source_test_loader_4, _ = data_loader(dataset_name=dataset_name, folder=source_folder_4,
                                                               batch_size=batch_size,
                                                               infinite_data_loader=infinite_data_loader,
                                                               train=None, num_workers=num_workers)

        return source_loader_1, source_loader_2, source_loader_3, \
               source_loader_4, target_loader, target_test_loader, n_class
    
    else:
        print('There is no dataset named ' + dataset_name)


if __name__ == '__main__':
    source_loader_1, source_loader_2, source_loader_3, \
    source_loader_4, source_loader_5, target_loader, n_class = data_read('ASD', 'LEUVEN', 32, False, True, 0)
    print(len(source_loader_1))
    print(len(source_loader_2))
    print(len(target_loader))
    print(n_class)
