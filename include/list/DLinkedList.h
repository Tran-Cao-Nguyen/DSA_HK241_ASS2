/*
 * File:   DLinkedList.h
 */

#ifndef DLINKEDLIST_H
#define DLINKEDLIST_H

#include "IList.h"

#include <sstream>
#include <iostream>
#include <type_traits>
using namespace std;

template <class T>
class DLinkedList : public IList<T>
{
public:
    class Node;        // Forward declaration
    class Iterator;    // Forward declaration
    class BWDIterator; // Forward declaration

protected:
    Node *head; // this node does not contain user's data
    Node *tail; // this node does not contain user's data
    int count;
    bool (*itemEqual)(T &lhs, T &rhs);        // function pointer: test if two items (type: T&) are equal or not
    void (*deleteUserData)(DLinkedList<T> *); // function pointer: be called to remove items (if they are pointer type)

public:
    DLinkedList(
        void (*deleteUserData)(DLinkedList<T> *) = 0,
        bool (*itemEqual)(T &, T &) = 0);
    DLinkedList(const DLinkedList<T> &list);
    DLinkedList<T> &operator=(const DLinkedList<T> &list);
    ~DLinkedList();

    // Inherit from IList: BEGIN
    void add(T e);
    void add(int index, T e);
    T removeAt(int index);
    bool removeItem(T item, void (*removeItemData)(T) = 0);
    bool empty();
    int size();
    void clear();
    T &get(int index);
    int indexOf(T item);
    bool contains(T item);
    string toString(string (*item2str)(T &) = 0);
    // Inherit from IList: END

    void println(string (*item2str)(T &) = 0)
    {
        cout << toString(item2str) << endl;
    }
    void setDeleteUserDataPtr(void (*deleteUserData)(DLinkedList<T> *) = 0)
    {
        this->deleteUserData = deleteUserData;
    }

    bool contains(T array[], int size)
    {
        int idx = 0;
        for (DLinkedList<T>::Iterator it = begin(); it != end(); it++)
        {
            if (!equals(*it, array[idx++], this->itemEqual))
                return false;
        }
        return true;
    }

    /*
     * free(DLinkedList<T> *list):
     *  + to remove user's data (type T, must be a pointer type, e.g.: int*, Point*)
     *  + if users want a DLinkedList removing their data,
     *      he/she must pass "free" to constructor of DLinkedList
     *      Example:
     *      DLinkedList<T> list(&DLinkedList<T>::free);
     */
    static void free(DLinkedList<T> *list)
    {
        typename DLinkedList<T>::Iterator it = list->begin();
        while (it != list->end())
        {
            delete *it;
            it++;
        }
    }

    /* begin, end and Iterator helps user to traverse a list forwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::Iterator it;
     for(it = list.begin(); it != list.end(); it++){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    Iterator begin()
    {
        return Iterator(this, true);
    }
    Iterator end()
    {
        return Iterator(this, false);
    }

    /* last, beforeFirst and BWDIterator helps user to traverse a list backwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::BWDIterator it;
     for(it = list.last(); it != list.beforeFirst(); it--){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    BWDIterator bbegin()
    {
        return BWDIterator(this, true);
    }
    BWDIterator bend()
    {
        return BWDIterator(this, false);
    }

protected:
    static bool equals(T &lhs, T &rhs, bool (*itemEqual)(T &, T &))
    {
        if (itemEqual == 0)
            return lhs == rhs;
        else
            return itemEqual(lhs, rhs);
    }
    void copyFrom(const DLinkedList<T> &list);
    void removeInternalData();
    Node *getPreviousNodeOf(int index);

    //////////////////////////////////////////////////////////////////////
    ////////////////////////  INNER CLASSES DEFNITION ////////////////////
    //////////////////////////////////////////////////////////////////////
public:
    class Node
    {
    public:
        T data;
        Node *next;
        Node *prev;
        friend class DLinkedList<T>;

    public:
        Node(Node *next = 0, Node *prev = 0)
        {
            this->next = next;
            this->prev = prev;
        }
        Node(T data, Node *next = 0, Node *prev = 0)
        {
            this->data = data;
            this->next = next;
            this->prev = prev;
        }
    };

    //////////////////////////////////////////////////////////////////////
    class Iterator
    {
    private:
        DLinkedList<T> *pList;
        Node *pNode;

    public:
        Iterator(DLinkedList<T> *pList = 0, bool begin = true)
        {
            if (begin)
            {
                if (pList != 0)
                    this->pNode = pList->head->next;
                else
                    pNode = 0;
            }
            else
            {
                if (pList != 0)
                    this->pNode = pList->tail;
                else
                    pNode = 0;
            }
            this->pList = pList;
        }

        Iterator &operator=(const Iterator &iterator)
        {
            this->pNode = iterator.pNode;
            this->pList = iterator.pList;
            return *this;
        }
        void remove(void (*removeItemData)(T) = 0)
        {
            pNode->prev->next = pNode->next;
            pNode->next->prev = pNode->prev;
            Node *pNext = pNode->prev; // MUST prev, so iterator++ will go to end
            if (removeItemData != 0)
                removeItemData(pNode->data);
            delete pNode;
            pNode = pNext;
            pList->count -= 1;
        }

        T &operator*()
        {
            return pNode->data;
        }
        bool operator!=(const Iterator &iterator)
        {
            return pNode != iterator.pNode;
        }
        // Prefix ++ overload
        Iterator &operator++()
        {
            pNode = pNode->next;
            return *this;
        }
        // Postfix ++ overload
        Iterator operator++(int)
        {
            Iterator iterator = *this;
            ++*this;
            return iterator;
        }
    };
};
//////////////////////////////////////////////////////////////////////
// Define a shorter name for DLinkedList:

template <class T>
using List = DLinkedList<T>;

//////////////////////////////////////////////////////////////////////
////////////////////////     METHOD DEFNITION      ///////////////////
//////////////////////////////////////////////////////////////////////

template <class T>
DLinkedList<T>::DLinkedList(
    void (*deleteUserData)(DLinkedList<T> *),
    bool (*itemEqual)(T &, T &))
{
    // TODO
    this->head = new Node();
    this->tail = new Node();

    this->head->next = this->tail;
    this->head->prev = nullptr;

    this->tail->next = nullptr;
    this->tail->prev = this->head;

    this->count = 0;
    this->deleteUserData = deleteUserData;
    this->itemEqual = itemEqual;
}

template <class T>
DLinkedList<T>::DLinkedList(const DLinkedList<T> &list)
{
    // TODO
    this->head = new Node();
    this->tail = new Node();

    this->copyFrom(list);
}

template <class T>
DLinkedList<T> &DLinkedList<T>::operator=(const DLinkedList<T> &list)
{
    // TODO
    this->copyFrom(list);
    return *this;
}

template <class T>
DLinkedList<T>::~DLinkedList()
{
    // TODO
    removeInternalData();

    if (head != 0)
        delete head;
    if (tail != 0)
        delete tail;
}

template <class T>
void DLinkedList<T>::add(T e)
{
    // TODO
    Node *newNode = new Node(e, this->tail, this->tail->prev);

    this->tail->prev->next = newNode;
    this->tail->prev = newNode;
    this->count++;
}
template <class T>
void DLinkedList<T>::add(int index, T e)
{
    // TODO
    if (index < 0 || index > this->count)
    {
        throw out_of_range("index is out of range");
    }

    if (index == 0)
    {
        Node *newNode = new Node(e, this->head->next, this->head);
        this->head->next->prev = newNode;
        this->head->next = newNode;
        this->tail->prev = newNode;
        this->count++;
    }
    else if (index == this->count)
    {
        this->add(e);
    }
    else
    {
        Node *temp = this->getPreviousNodeOf(index)->next;
        Node *prevPtr = temp->prev;

        Node *newNode = new Node(e, temp, prevPtr);

        prevPtr->next = newNode;
        temp->prev = newNode;
        this->count++;
    }
}

template <class T>
typename DLinkedList<T>::Node *DLinkedList<T>::getPreviousNodeOf(int index)
{
    /**
     * Returns the node preceding the specified index in the doubly linked list.
     * If the index is in the first half of the list, it traverses from the head; otherwise, it traverses from the tail.
     * Efficiently navigates to the node by choosing the shorter path based on the index's position.
     */
    // TODO

    if (index <= 0 || index >= count)
    {
        throw out_of_range("index is out of range");
    }
    Node *cur;
    if (index < this->count / 2)
    {
        cur = this->head->next;
        for (int i = 0; i < index; i++)
        {
            cur = cur->next;
        }
    }
    else
    {
        cur = this->tail->prev;
        for (int i = this->count - 1; i > index; i--)
        {
            cur = cur->prev;
        }
    }
    return cur->prev;
}

template <class T>
T DLinkedList<T>::removeAt(int index)
{
    // TODO
    if (index < 0 || index >= this->count)
    {
        throw out_of_range("index is out of range");
    }

    if (index == 0)
    {
        Node *cur = this->head->next;
        T returnData = cur->data;

        this->head->next = cur->next;
        cur->next->prev = this->head;

        delete cur;

        return returnData;
    }
    else if (index == this->count - 1)
    {
        Node *cur = this->tail->prev;
        T returnData = cur->data;

        this->tail->prev = cur->prev;
        cur->prev->next = this->tail;

        delete cur;

        return returnData;
    }
    else
    {
        Node *preNode = this->getPreviousNodeOf(index);
        Node *cur = preNode->next;
        Node *nextNode = cur->next;

        T returnData = cur->data;

        preNode->next = nextNode;
        nextNode->prev = preNode;

        delete cur;

        return returnData;
    }
}

template <class T>
bool DLinkedList<T>::empty()
{
    // TODO
    return this->count == 0;
}

template <class T>
int DLinkedList<T>::size()
{
    // TODO
    return this->count;
}

template <class T>
void DLinkedList<T>::clear()
{
    // TODO
    this->removeInternalData();

    this->head->next = this->tail;
    this->head->prev = nullptr;

    this->tail->next = nullptr;
    this->tail->prev = this->head;

    this->count = 0;
}

template <class T>
T &DLinkedList<T>::get(int index)
{
    // TODO
    if (index < 0 || index >= count)
    {
        throw out_of_range("index is out of range");
    }

    Node *cur;
    if (index < this->count / 2)
    {
        cur = this->head->next;
        for (int i = 0; i < index; i++)
        {
            cur = cur->next;
        }
    }
    else
    {
        cur = this->tail->prev;
        for (int i = this->count - 1; i > index; i--)
        {
            cur = cur->prev;
        }
    }
    return cur->data;
}

template <class T>
int DLinkedList<T>::indexOf(T item)
{
    // TODO
    Node *cur = this->head->next;
    int index = 0;
    while (cur != tail)
    {
        if (this->equals(cur->data, item, this->itemEqual))
        {
            return index;
        }
        index++;
        cur = cur->next;
    }
    return -1;
}

template <class T>
bool DLinkedList<T>::removeItem(T item, void (*removeItemData)(T))
{
    // TODO
    // if (!this->empty())
    // {
    //     Node *cur = this->head->next;
    //     int index = 0;
    //     while (cur != tail)
    //     {
    //         if (this->equals(cur->data, item, nullptr))
    //         {
    //             if (removeItemData)
    //             {
    //                 removeItemData(cur->data);
    //             }
    //             this->removeAt(index);
    //             return true;
    //         }
    //         else
    //         {
    //             index++;
    //             cur = cur->next;
    //         }
    //     }
    // }
    // return false;

    int index = this->indexOf(item);
    if (index != -1)
    {
        T removeData = this->removeAt(index);
        if (removeData == item)
        {
            if (removeItemData)
            {
                removeItemData(removeData);
            }
            return true;
        }
    }
    return false;
}

template <class T>
bool DLinkedList<T>::contains(T item)
{
    // TODO
    Node *cur = this->head->next;
    while (cur != tail)
    {
        if (this->equals(cur->data, item, this->itemEqual))
        {
            return true;
        }
        cur = cur->next;
    }
    return false;
}

template <class T>
string DLinkedList<T>::toString(string (*item2str)(T &))
{
    /**
     * Converts the list into a string representation, where each element is formatted using a user-provided function.
     * If no custom function is provided, it directly uses the element's default string representation.
     * Example: If the list contains {1, 2, 3} and the provided function formats integers, calling toString would return "[1, 2, 3]".
     *
     * @param item2str A function that converts an item of type T to a string. If null, default to string conversion of T.
     * @return A string representation of the list with elements separated by commas and enclosed in square brackets.
     */
    // TODO

    std::stringstream ss;
    ss << "[";
    Node *cur = this->head->next;
    int i = 0;
    while (cur != this->tail)
    {
        if (i > 0 && i < this->count)
        {
            ss << ", ";
        }

        if (item2str)
        {
            ss << item2str(cur->data);
        }
        else
        {
            ss << cur->data;
        }
        i++;
        cur = cur->next;
    }

    ss << "]";
    return ss.str();
}

template <class T>
void DLinkedList<T>::copyFrom(const DLinkedList<T> &list)
{
    /**
     * Copies the contents of another doubly linked list into this list.
     * Initializes the current list to an empty state and then duplicates all data and pointers from the source list.
     * Iterates through the source list and adds each element, preserving the order of the nodes.
     */
    // TODO

    this->count = list.count;
    this->deleteUserData = list.deleteUserData;
    this->itemEqual = list.itemEqual;

    this->removeInternalData();

    this->head->next = this->tail;
    this->head->prev = nullptr;

    this->tail->next = nullptr;
    this->tail->prev = this->head;

    if (list.head != nullptr)
    {
        Node *cur = list.head->next;
        while (cur != list.tail)
        {
            this->add(cur->data);
            cur = cur->next;
        }
    }
}

template <class T>
void DLinkedList<T>::removeInternalData()
{
    /**
     * Clears the internal data of the list by deleting all nodes and user-defined data.
     * If a custom deletion function is provided, it is used to free the user's data stored in the nodes.
     * Traverses and deletes each node between the head and tail to release memory.
     */
    // TODO
    if (this->empty())
        return;
    Node *cur = this->head->next;
    while (cur != this->tail)
    {
        Node *next = cur->next;
        delete cur;
        cur = next;
    }
}

#endif /* DLINKEDLIST_H */
