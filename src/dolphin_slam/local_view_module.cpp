#include "local_view_module.h"

const float ROS_TIMER_STEP = 0.25;

using std::endl;
using std::cout;

namespace dolphin_slam
{

LocalViewModule::LocalViewModule()
{

    metrics_.creation_count_ = 0;
    metrics_.recognition_count_ = 0;

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

}

void LocalViewModule::createROSSubscribers()
{
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



void LocalViewModule::descriptors_callback(const DescriptorsConstPtr &msg)
{


    ROS_DEBUG_STREAM("Descriptors received. seq = " << msg->image_seq_ << " Number of descriptors = "  << msg->descriptor_count_);


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


void  LocalViewModule::computeMatches()
{
    if (cells_.size() == 0)
    {
        new_place_ = true;
    }
    else
    {
        if(parameters_.matching_algorithm_ == "correlation")
        {
            computeCorrelations();
        }
        else
        {
            ROS_ERROR_STREAM("Matching algorithm is wrong.");
            exit(0);
        }
    }

    if(new_place_)
    {
        createNewCell();
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
    }

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



} //namespace
