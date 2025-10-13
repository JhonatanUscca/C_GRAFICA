#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <chrono> 

struct Node {
    int val;
    Node* next;
    Node(int v): val(v), next(nullptr) {}
};

Node* head = nullptr;
pthread_rwlock_t list_lock; 


bool Member(int value) {
    bool found = false;
    pthread_rwlock_rdlock(&list_lock); 
    Node* curr = head;
    while (curr && curr->val < value) curr = curr->next;
    if (curr && curr->val == value) found = true;
    pthread_rwlock_unlock(&list_lock);
    return found;
}


bool Insert(int value) {
    pthread_rwlock_wrlock(&list_lock); 
    Node* prev = nullptr;
    Node* curr = head;
    while (curr && curr->val < value) {
        prev = curr;
        curr = curr->next;
    }
    if (curr && curr->val == value) {
        pthread_rwlock_unlock(&list_lock);
        return false;
    }
    Node* newn = new Node(value);
    newn->next = curr;
    if (!prev) head = newn;
    else prev->next = newn;
    pthread_rwlock_unlock(&list_lock);
    return true;
}


bool Delete(int value) {
    pthread_rwlock_wrlock(&list_lock); 
    Node* prev = nullptr;
    Node* curr = head;
    while (curr && curr->val < value) {
        prev = curr;
        curr = curr->next;
    }
    if (!curr || curr->val != value) {
        pthread_rwlock_unlock(&list_lock);
        return false; 
    }
    if (!prev) head = curr->next;
    else prev->next = curr->next;
    delete curr;
    pthread_rwlock_unlock(&list_lock);
    return true;
}


struct ThreadArg {
    int id;      
    int ops;     
};


void* thread_main(void* a) {
    ThreadArg* arg = (ThreadArg*)a;
    unsigned int seed = (unsigned int)time(nullptr) ^ arg->id;  
    int total_ops = arg->ops;

    int member_ops = total_ops * 80 / 100;  
    int insert_ops = total_ops * 10 / 100;  
    int delete_ops = total_ops * 10 / 100;  

   
    for (int i = 0; i < member_ops; i++) {
        int v = rand_r(&seed) % 1000;  
        Member(v);
    }

    
    for (int i = 0; i < insert_ops; i++) {
        int v = rand_r(&seed) % 1000;
        Insert(v);
    }

   
    for (int i = 0; i < delete_ops; i++) {
        int v = rand_r(&seed) % 1000;
        Delete(v);
    }

    return nullptr;
}

int main(int argc, char** argv) {
    int nthreads[] = {1, 2, 4, 8};  
    int ops_per_thread = 100000;

    
    pthread_rwlock_init(&list_lock, nullptr);

    for (int i = 0; i < 4; i++) {
        int nthread = nthreads[i];
        std::cout << "Evaluando con " << nthread << " hilos\n";

        
        auto start = std::chrono::high_resolution_clock::now();

        pthread_t threads[nthread];
        ThreadArg args[nthread];

        
        for (int j = 0; j < nthread; j++) {
            args[j].id = j;
            args[j].ops = ops_per_thread;
            pthread_create(&threads[j], nullptr, thread_main, &args[j]);
        }

        
        for (int j = 0; j < nthread; j++) {
            pthread_join(threads[j], nullptr);
        }

       
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Tiempo de ejecucion con " << nthread << " hilos: " 
                  << duration.count() << " segundos.\n\n";

        
        pthread_rwlock_wrlock(&list_lock);
        while (head) {
            Node* t = head;
            head = head->next;
            delete t;
        }
        pthread_rwlock_unlock(&list_lock);
    }

    pthread_rwlock_destroy(&list_lock);

    return 0;
}
