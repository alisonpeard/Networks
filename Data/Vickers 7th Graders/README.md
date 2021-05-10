

# VICKERS-CHAN-7THGRADERS MULTIPLEX NETWORK

###### Last update: 1 July 2014

### Reference and Acknowledgments

This README file accompanies the dataset representing the multiplex social network in a school in Victoria, Australia.
If you use this dataset in your work either for analysis or for visualization, you should acknowledge/cite the following papers:

	Representing Classroom Social Structure. Melbourne: Victoria Institute of Secondary Education 
	M. Vickers and S. Chan, (1981)


### Description of the dataset

The data were collected by Vickers from 29 seventh grade students in a school in Victoria, Australia. Students were asked to nominate their classmates on a number of relations including the following three (layers):

1. Who do you get on with in the class?
2. Who are your best friends in the class?
3. Who would you prefer to work with?

Students 1 through 12 are boys and 13 through 29 are girls.

There are 29 nodes in total, labelled with integer ID between 1 and 29, with 740 connections.
The multiplex is directed and unweighted, stored as edges list in the file
    
    Vickers-Chan-7thGraders_multiplex.edges

with format

    layerID nodeID nodeID weight

(Note: all weights are set to 1)

The IDs of all layers are stored in 

    Vickers-Chan-7thGraders_layers.txt


### License

The VICKERS-CHAN-7THGRADERS MULTIPLEX DATASET is provided "as is" and without warranties as to performance or quality or any other warranties whether expressed or implied. 

