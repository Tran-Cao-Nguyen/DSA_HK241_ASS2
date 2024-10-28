/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "tensor/xtensor_lib.h"
#include "loader/dataset.h"

using namespace std;

template <typename DType, typename LType>
class DataLoader
{
public:
    // class Iterator
    class Iterator;

private:
    Dataset<DType, LType> *ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    int m_seed;
    /*TODO: add more member variables to support the iteration*/

    xt::xarray<unsigned long> indices; // Danh mục chỉ số
    unsigned long current_index;       // Chỉ số hiện tại trong iterator

public:
    DataLoader(Dataset<DType, LType> *ptr_dataset,
               int batch_size,
               bool shuffle = true,
               bool drop_last = false, int seed = -1)
    {
        /*TODO: Add your code to do the initialization */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->m_seed = seed;

        indices = xt::arange<unsigned long>(0, ptr_dataset->len());

        if (this->shuffle)
        {
            if (this->m_seed >= 0)
            {
                xt::random::seed(this->m_seed);
            }
            xt::random::shuffle(this->indices);
        }

        current_index = 0;
    }
    virtual ~DataLoader() {}

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    /*TODO: Add your code here to support iteration on batch*/

    Iterator begin()
    {
        return Iterator(this, 0);
    }

    Iterator end()
    {
        unsigned long num_batches = ptr_dataset->len() / batch_size;
        return Iterator(this, num_batches);
    }
    class Iterator
    {
    private:
        DataLoader<DType, LType> *loader;
        unsigned long index;

    public:
        Iterator(DataLoader<DType, LType> *loader, unsigned long index)
        {
            this->loader = loader;
            this->index = index;
        }

        bool operator!=(const Iterator &other) const
        {
            return this->index != other.index;
        }

        Batch<DType, LType> operator*()
        {
            unsigned long start = this->index * loader->batch_size;
            unsigned long end;
            if ((this->index + 1) * loader->batch_size + loader->batch_size >= loader->indices.size() && !loader->drop_last)
            {
                end = static_cast<unsigned long>(loader->indices.size());
            }
            else
            {
                end = start + loader->batch_size;
            }

            int data_dimensions = loader->ptr_dataset->get_data_dimension();
            int label_dimensions = loader->ptr_dataset->get_label_dimension();

            vector<std::size_t> batch_shape(data_dimensions, 0);

            batch_shape[0] = end - start;
            for (int i = 1; i < data_dimensions; i++)
            {
                batch_shape[i] = loader->ptr_dataset->get_data_shape()[i];
            }
            xt::xarray<DType> batch_data = xt::empty<DType>(batch_shape);

            vector<std::size_t> label_shape(label_dimensions, 0);
            xt::xarray<LType> batch_labels;
            if (label_dimensions > 0)
            {

                label_shape[0] = end - start;
                for (int i = 1; i < label_dimensions; i++)
                {
                    label_shape[i] = loader->ptr_dataset->get_label_shape()[i];
                }
                batch_labels = xt::empty<LType>(label_shape);
            }

            for (unsigned long i = start; i < end; i++)
            {
                unsigned long data_index = loader->indices(i);

                xt::view(batch_data, i - start) = loader->ptr_dataset->getitem(data_index).getData();
                if (label_dimensions > 0)
                {
                    xt::view(batch_labels, i - start) = loader->ptr_dataset->getitem(data_index).getLabel();
                }
            }
            if (label_dimensions > 0)
                return Batch<DType, LType>(batch_data, batch_labels);
            else
                return Batch<DType, LType>(batch_data, xt::xarray<LType>());
        }
        // Prefix
        Iterator &operator++()
        {
            ++index;

            return *this;
        }
        // Postfix
        Iterator operator++(int)
        {
            Iterator it = *this;
            ++*this;
            return it;
        }
    };

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};

#endif /* DATALOADER_H */
