#include "local_view_module.h"
#include "dolphin_slam/ImgMatch.h"
#include "dolphin_slam/ImgMatchArray.h"
const float ROS_TIMER_STEP = 0.25;

using std::endl;
using std::cout;

namespace dolphin_slam
{

LocalViewModule::LocalViewModule()
{
    ROS_DEBUG_STREAM("LOCALVIEW");
    metrics_.creation_count_ = 0;
    metrics_.recognition_count_ = 0;
    matchs_others.reserve(7000);
    loadParameters();

    createROSSubscribers();

    createROSPublishers();

    init();

    log_file_rate_.open("localviews_rate.log");
    log_file_metrics_.open("localviews_metrics.log");
    log_file_bow_.open("bow_descriptors.txt");

}

LocalViewModule::~LocalViewModule()
{
    log_file_rate_.close();
    log_file_metrics_.close();
    log_file_bow_.close();
}

void LocalViewModule::loadParameters()
{
    ros::NodeHandle private_nh("~");

    private_nh.param<double>("similarity_threshold",parameters_.similarity_threshold_,0.85);

    private_nh.param<std::string>("matching_algorithm",parameters_.matching_algorithm_,"correlation");

    private_nh.param<std::string>("descriptors_topic",parameters_.descriptors_topic_,"/descriptors");

    private_nh.param<std::string>("bow_vocab",parameters_.bow_vocab_,"vocabulary.xml");

    private_nh.param<std::string>("type_algorithm",parameters_.type_algorithm,"normal");

}

void LocalViewModule::createROSSubscribers()
{

     gt_loop_ = node_handle_.subscribe("/gt_loop",1,&LocalViewModule::gt_loop,this);

    descriptors_subscriber_ = node_handle_.subscribe(parameters_.descriptors_topic_,1,&LocalViewModule::descriptors_callback,this);

}

void LocalViewModule::createROSPublishers()
{
    active_cells_publisher_ = node_handle_.advertise<dolphin_slam::ActiveLocalViewCells>("local_view_cells",1);

    execution_time_publisher_ = node_handle_.advertise<dolphin_slam::ExecutionTime>("execution_time",1,false);
}


void LocalViewModule::init()
{
    cv::FileStorage fs;
    
    if (parameters_.matching_algorithm_== "correlation")
    {
        fs.open(parameters_.bow_vocab_,cv::FileStorage::READ);
        fs["vocabulary"] >> bow_vocabulary_;
        fs.release();

        bow_extractor_ = new BOWImgDescriptorExtractor(cv::DescriptorMatcher::create("FlannBased"));
        bow_extractor_->setVocabulary(bow_vocabulary_);

    }
    else
    {
        ROS_ERROR("Invalid matching algorithm");
    }

    //! Estudar se tem mais coisas a adicionar na inicialização


}


void LocalViewModule::createROSTimers()
{
    timer_ = node_handle_.createTimer(ros::Duration(0.5), &LocalViewModule::timerCallback,this);

}

void LocalViewModule::timerCallback(const ros::TimerEvent& event)
{

}


void LocalViewModule::gt_loop(const dolphin_slam::ImgMatchArrayConstPtr &msg)
{
    if(parameters_.type_algorithm =="others"){

        ROS_DEBUG_STREAM("1 srcID= " << msg->srcID );

        /*imgMatch.resize(msg->match.size());
       / for(unsigned i = 0 ; i < msg->match.size();i++)
        {
            imgMatch[i] = msg->match[i].dstID;
        }*/

       //Se vier um vetor de match, pega a primeira posição do vetor match
        if(msg->match.size()>0)
        {

            gt_id_=msg->match[0].dstID;
            ROS_DEBUG_STREAM("1 gt_id = " << gt_id_);
            gt_src_=msg->srcID;
        }
        else
        {
            //Se não vier vetor, define a variável de gt_id
            gt_id_=-1;
        }

        if(metrics_.creation_count_ == 0)
        {
            start_stamp_ = ros::Time::now();
        }

        time_monitor_.start();

        last_best_match_id_ = best_match_id_;


        //ESCREVER FUNçAO DE MATCH
        computeMatches();

        time_monitor_.finish();

        publishActiveCells1();

        //! Compute metrics
        if(new_place_)
        {
            metrics_.creation_count_++;
        }
        else
        {
            if(best_match_id_ != last_best_match_id_)
            {
                metrics_.recognition_count_++;
            }
        }
        metrics_.execution_time_ = time_monitor_.getDuration();

        //writeLog();

        publishExecutionTime();

    }
}

void LocalViewModule::descriptors_callback(const DescriptorsConstPtr &msg)
{

    if(parameters_.type_algorithm == "normal")
    {

        ROS_DEBUG_STREAM("NORMAL - Descriptors received. seq = " << msg->image_seq_ << " Number of descriptors = "  << msg->descriptor_count_);


        if(metrics_.creation_count_ == 0)
        {
            start_stamp_ = ros::Time::now();
        }

        time_monitor_.start();

        last_best_match_id_ = best_match_id_;

        image_seq_ = msg->image_seq_;
        image_stamp_ = msg->image_stamp_;

        // Copy descriptors into a cv::Mat
        cv::Mat_<float> descriptors(msg->descriptor_count_,msg->descriptor_length_);
        std::copy(msg->data_.begin(),msg->data_.end(),descriptors.begin());


        computeImgDescriptor(descriptors);

        cout << bow_current_descriptor_ << endl;

        computeMatches();


        time_monitor_.finish();

        publishActiveCells();

        //! Compute metrics
        if(new_place_)
        {
            metrics_.creation_count_++;
        }
        else
        {
            if(best_match_id_ != last_best_match_id_)
            {
                metrics_.recognition_count_++;
            }
        }
        metrics_.execution_time_ = time_monitor_.getDuration();

        writeLog();

        publishExecutionTime();
    }

}


void  LocalViewModule::computeMatches()
{
    if (cells_.size() == 0)
    {
        new_place_ = true;
        matchs_others.push_back(0);
    }
    else
    {
        if(parameters_.matching_algorithm_ == "correlation")
        {
            if(parameters_.type_algorithm == "normal")
            {
              computeCorrelations();
            }
            else if(parameters_.type_algorithm == "others")
            {
              computeCorrelations1();
            }
        }
        else
        {
            ROS_ERROR_STREAM("Matching algorithm is wrong.");
            exit(0);
        }
    }

    if(new_place_)
    {
        if(parameters_.type_algorithm == "normal")
        {
          createNewCell();
        }
        else if(parameters_.type_algorithm == "others")
        {
          createNewCell1();
        }
    }
}

void LocalViewModule::writeLog()
{
    double stamp = (ros::Time::now() - start_stamp_).toSec();
    log_file_rate_ << stamp << " ";

    std::vector<LocalViewCell>::iterator cell_iterator_;
    for(cell_iterator_ = cells_.begin();cell_iterator_!= cells_.end();cell_iterator_++)
    {
        log_file_rate_ << cell_iterator_->rate_ << " " ;
    }
    log_file_rate_ << std::endl;

    log_file_metrics_ << stamp << " "
                      << best_match_id_ << " "
                      << metrics_.execution_time_ << " "
                      << metrics_.creation_count_ << " "
                      << metrics_.recognition_count_ << " "
                      << std::endl;

    log_file_bow_ << image_seq_ << " ";

    for(int i=0;i<bow_current_descriptor_.cols;i++)
    {
        log_file_bow_ << bow_current_descriptor_.at<float>(0,i)<< " ";
    
    }
    log_file_bow_ << endl;

}


void LocalViewModule::publishExecutionTime()
{
    ExecutionTime msg;
    msg.header.stamp = ros::Time::now();

    msg.module = "lv";
    msg.iteration_time = time_monitor_.getDuration();

    execution_time_publisher_.publish(msg);

}

void LocalViewModule::computeCorrelations1()
{
    new_place_ = true;
    std::vector<LocalViewCell>::iterator cell_iterator_;
    std::vector<LocalViewCell>::iterator best_match;
  //          best_match = cells_.begin();

    if (gt_id_ == -1)
    {
       new_place_ = true;
       matchs_others.push_back(cells_.size());
      //ROS_DEBUG_STREAM("2 match[gt_src_] " << matchs_others[gt_src_]);
    }

    else
    {
        //ROS_DEBUG_STREAM("3 gt_id_ = " << gt_id_);
        best_match = cells_.begin()+matchs_others[gt_id_];
        new_place_ = false;
        ROS_DEBUG_STREAM("3 best_match_id = " << best_match->id_);
        matchs_others.push_back(matchs_others[gt_id_]);
        for(int i=0; i < cells_.size(); i++)
        {
            cells_[i].active_ = false;
            cells_[i].rate_ = 0;

        }
       cells_[matchs_others[gt_id_]].active_ = true;
       cells_[matchs_others[gt_id_]].rate_ = 1;
       ROS_DEBUG_STREAM("3 matchs_others[gt_src_] = " << matchs_others[gt_id_]);
    }

    if(!new_place_)
    {
        new_rate_ = 0;
        best_match_id_ = best_match->id_;
        // ROS_DEBUG_STREAM("4 Best_match_id = " << best_match_id_);
        /* for (int j=0; j < gt_src_; j++)
         {
            ROS_DEBUG_STREAM("vetor[" << j << "] , valor = "<< matchs_others[j]);
         }*/
    }
    else
    {
        new_rate_ = 1;
    }
}

void LocalViewModule::computeCorrelations()
{
    new_place_ = true;
    std::vector<LocalViewCell>::iterator cell_iterator_;
    std::vector<LocalViewCell>::iterator best_match;
    best_match = cells_.begin();


        cout << "correlations ";
        for(cell_iterator_ = cells_.begin();cell_iterator_!= cells_.end();cell_iterator_++)
        {
            cell_iterator_->rate_ = cv::compareHist(bow_descriptors_[cell_iterator_->id_],bow_current_descriptor_,CV_COMP_CORREL);

            cout << cell_iterator_->rate_ << " " ;
            //! test activation against a similarity threshold;
            cell_iterator_->active_ = (cell_iterator_->rate_ > parameters_.similarity_threshold_);

            if(cell_iterator_->active_)
            {
                new_place_ = false;
            }

            //! compute best match
            if(best_match->rate_ < cell_iterator_->rate_)
            {
                best_match = cell_iterator_;
            }
        }//fim do for

        cout << endl;

    if(!new_place_)
    {
        new_rate_ = 0;
        best_match_id_ = best_match->id_;
    }
    else
    {
        new_rate_ = 1;
    }

}


void LocalViewModule::createNewCell()
{
    LocalViewCell new_cell;

    new_cell.id_ = cells_.size();
    new_cell.rate_ = new_rate_;
    new_cell.active_ = true;

    best_match_id_ = new_cell.id_;

    bow_descriptors_.push_back(bow_current_descriptor_);

    cells_.push_back(new_cell);

}

void LocalViewModule::createNewCell1()
{
    LocalViewCell new_cell;

    new_cell.id_ = cells_.size();
    new_cell.rate_ = new_rate_;
    new_cell.active_ = true;

    best_match_id_ = new_cell.id_;

    //bow_descriptors_.push_back(bow_current_descriptor_);

    cells_.push_back(new_cell);

}



void LocalViewModule::computeImgDescriptor(cv::Mat & descriptors)
{
    if (parameters_.matching_algorithm_ == "correlation")
    {
        bow_extractor_->compute(descriptors,bow_current_descriptor_);
    }
}


void LocalViewModule::publishActiveCells(){

    ActiveLocalViewCells msg;

    msg.header.stamp = ros::Time::now();

    msg.image_seq_ = image_seq_;
    msg.image_stamp_ = image_stamp_;

    msg.most_active_cell_ = best_match_id_;

    if(parameters_.matching_algorithm_ == "correlation")
    {
        log_file_rate_ << (ros::Time::now() - start_stamp_).toSec() << " ";

        std::vector<LocalViewCell>::iterator cell_iterator_;
        for(cell_iterator_ = cells_.begin();cell_iterator_!= cells_.end();cell_iterator_++)
        {
            if(cell_iterator_->active_)
            {
                msg.cell_id_.push_back(cell_iterator_->id_);
                msg.cell_rate_.push_back(cell_iterator_->rate_);
            }
        }
    }
    else
    {
        ROS_ERROR_STREAM("Wrong matching algorithm: " << parameters_.matching_algorithm_);
    }

    active_cells_publisher_.publish(msg);

}

void LocalViewModule::publishActiveCells1(){

    ActiveLocalViewCells msg;

    msg.header.stamp = ros::Time::now();

    msg.image_seq_ = gt_src_;
   msg.image_stamp_ = image_stamp_;

    msg.most_active_cell_ = best_match_id_;

    if(parameters_.matching_algorithm_ == "correlation")
    {
        log_file_rate_ << (ros::Time::now() - start_stamp_).toSec() << " ";

        std::vector<LocalViewCell>::iterator cell_iterator_;
        for(cell_iterator_ = cells_.begin();cell_iterator_!= cells_.end();cell_iterator_++)
        {
            if(cell_iterator_->active_)
            {
                msg.cell_id_.push_back(cell_iterator_->id_);
                msg.cell_rate_.push_back(cell_iterator_->rate_);
            }
        }
    }
    else
    {
        ROS_ERROR_STREAM("Wrong matching algorithm: " << parameters_.matching_algorithm_);
    }

    active_cells_publisher_.publish(msg);

}


} //namespace
