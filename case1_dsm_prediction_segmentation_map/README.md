# Case 1: Topcoder Urban Mapper 3D dataset with DSM prediction

Download dataset: [Topcoder Urban Mapper 3D](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&compid=57607) 

` wget http://www.topcoder.com/contest/problem/UrbanMapper3D/training.zip `

Change path in DatasetHandler.py accordingly.
In train.py choose if we care about the DSM or the semantic segmentation (in this case we have just 2 reasonable classes - building and not-building)

run:  ` python3 train.py `

then you can use the trained model to predict with: ` python3 inference.py `

or: ` python3 inference_from_outside_data.py `


