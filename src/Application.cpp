#include "Application.h"
#include "cnn/ConvLayer.h"
#include "cnn/MaxPoolLayer.h"

#define LOAD_PATIENTS 0 // Flag to enable (1) or disable (0) loading patient data for debugging

Application* Application::m_Instance = nullptr;

Application::Application()
    : m_Running(false), m_InputMatrix(nullptr)
{
    Application::m_Instance = this;
}

Application::~Application()
{
    // Close the Log when the application ends
    MIR::Log::close();
}
void Application::buildNetwork()
{

    // Create a Tensor of the input matrix
    TensorPtrs* inputTensor = new TensorPtrs(m_InputMatrix);

    // Get input Data
    m_Network.init(inputTensor);

    // Create network
    // Layer 1
    ConvLayer* l1 = new ConvLayer("conv_layer_1", "sigmoid");
    l1->setNumNodes(3);

    // Node 1 random values between 0.0f and 1.0f
    Matrix* filter = new Matrix(4, 4, true, 0.0f, 1.0f);
    MIR::Log::writefInfo("Application::buildNetwork()", "Node 1 Random Weights: \n\n%s", Matrix::toString(*filter).c_str());

    // Node 2 random values between 0.0f and 1.0f
    Matrix* filter2 = new Matrix(4, 4, true, 0.0f, 1.0f);
    MIR::Log::writefInfo("Application::buildNetwork()", "Node 2 Random Weights: \n\n%s", Matrix::toString(*filter).c_str());

    // Node 3 random values between 0.0f and 1.0f
    Matrix* filter3 = new Matrix(4, 4, true, 0.0f, 1.0f);
    MIR::Log::writefInfo("Application::buildNetwork()", "Node 3 Random Weights: \n\n%s", Matrix::toString(*filter).c_str());

    l1->addNode(filter);
    l1->addNode(filter2);
    l1->addNode(filter3);

    // Layer.init
    l1->init(inputTensor, 4, 4);

    // Add Layer 1 to network
    m_Network.addLayer(std::move(l1));

    // Layer 2
    ConvLayer* l2 = new ConvLayer("conv_layer_2", "sigmoid");
    l2->setNumNodes(2);

    // Node 1
    Matrix* filter4 = new Matrix(3, 3, 1.5);

    // Node 2
    Matrix* filter5 = new Matrix(3, 3, 1.2);
    l2->addNode(filter4);
    l2->addNode(filter5);

    l2->init(l1->getOutput(), 3, 3);

    // Add Layer 2 to network
    m_Network.addLayer(std::move(l2));

    // Layer 3
    ConvLayer* l3 = new ConvLayer("conv_layer_3", "relu");
    l3->setNumNodes(1);

    // Node 1
    Matrix* filter6 = new Matrix(3, 3, 2.0);
    l3->addNode(filter6);

    l3->init(l2->getOutput(), 3, 3);

    // Add Layer 3 to network
    m_Network.addLayer(std::move(l3));

    // Max Pooling layer
    MaxPoolLayer* m1 = new MaxPoolLayer("max_pool_1", 2, false);
    m1->init(l3->getOutput());

    m_Network.addLayer(std::move(m1));
}


void Application::init()
{
    // Create a Log
    MIR::Log::create("V.0.1", "mirai_log.txt");
    MIR::Log::writeInfo("Application::init()", "Log created successfully");


    //image setup, firstInputMatrix is greyscale image data input ready for processing/iteration
    uint16_t width = 0;
    uint16_t height = 0;
    std::vector<unsigned char> vec = ImageWrapper::decodeImageToRGBAVector("res/images/00000001_000.png", width, height);
    std::vector<pixel> pix = ImageWrapper::convertRGBAVectorToPixelVector(vec);
    std::vector<double> doubs = ImageWrapper::convertPixelVectorToGreyscaleVector(pix);

    m_InputMatrix = new Matrix(1024, 1024, doubs);

    // Initialize Patient Handler 
#if LOAD_PATIENTS 
    MIR::Log::writeInfo("Application::init()", "Attempting to load patient data...");
    // Load and Construct Patient Population:
    if (!m_PatientHandler.load("res/test-data/test.csv"))
    {
        MIR::Log::writeEr("Application::init()", "Patient Loading failed, terminating program...");
        exitFailure();
    }
#else
    MIR::Log::writeWarn("Application::init()", "Patient Loading disabled, to renable set LOAD_PATIENTS to 1 in Application.cpp");
#endif

    // Start the main application loop
    m_Running = true;

    buildNetwork();
}




	
void Application::update()
{
	// Update program logic here

    m_Network.update();
    if (m_Network.isDone())
    {
        quit();
    }

    // Just set running to false to close the program for now
    //quit();
}

void Application::render()
{
    // We will draw stuff here maybe
}

void Application::run()
{
    // Initialize our application
    init();

    // Loop until we set m_Running to false
    while(m_Running)
    {
        update();
        
        //render(); No rendering currently
    }
}

void Application::exitFailure(const char* err)
{
    MIR::Log::writefEr("Application::exitFailure()", "%s", err);
    // Clean up the log if possible
    MIR::Log::close();

    exit(EXIT_FAILURE);
}

Application* Application::getInstance()
{
    return Application::m_Instance;
}

void Application::quit()
{
    Application::getInstance()->m_Running = false;
}
