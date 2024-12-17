import React, { useState } from 'react';
import { Upload, Video, Type, Layout, Image, BarChart2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Dashboard = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [metrics, setMetrics] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setMetrics({
        duration: "5:30",
        engagement: "85%",
        retention: "72%",
        viralPotential: "medium"
      });
    }, 2000);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Video Content Studio</h1>
        <p className="text-gray-600">Upload your video to access powerful content optimization tools</p>
      </div>

      {!selectedFile ? (
        <Card className="border-dashed border-2 cursor-pointer hover:bg-gray-50">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Upload className="w-12 h-12 text-gray-400 mb-4" />
            <label className="cursor-pointer">
              <span className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                Choose Video
              </span>
              <input
                type="file"
                className="hidden"
                accept="video/*"
                onChange={handleFileUpload}
              />
            </label>
            <p className="text-sm text-gray-500 mt-2">or drag and drop your video file</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Duration</CardTitle>
                <Video className="w-4 h-4 text-gray-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.duration}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Engagement</CardTitle>
                <BarChart2 className="w-4 h-4 text-gray-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.engagement}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Retention</CardTitle>
                <Layout className="w-4 h-4 text-gray-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.retention}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Viral Potential</CardTitle>
                <BarChart2 className="w-4 h-4 text-gray-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold capitalize">{metrics?.viralPotential}</div>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="storyboard" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="storyboard">Storyboard</TabsTrigger>
              <TabsTrigger value="thumbnail">Thumbnail Generator</TabsTrigger>
              <TabsTrigger value="captions">Caption Generator</TabsTrigger>
            </TabsList>
            <TabsContent value="storyboard">
              <Card>
                <CardHeader>
                  <CardTitle>Video Storyboard</CardTitle>
                  <CardDescription>Optimize your content structure and timing</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-4 gap-4">
                      {[1, 2, 3, 4].map((point) => (
                        <div key={point} className="border rounded-lg p-4">
                          <h3 className="font-medium mb-2">Engagement Point {point}</h3>
                          <p className="text-sm text-gray-600">Add hook for viewer retention</p>
                          <p className="text-sm text-gray-500 mt-2">Timing: {point}:00</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="thumbnail">
              <Card>
                <CardHeader>
                  <CardTitle>Thumbnail Generator</CardTitle>
                  <CardDescription>Create eye-catching thumbnails</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    {[1, 2, 3].map((template) => (
                      <div key={template} className="border rounded-lg p-4">
                        <div className="aspect-video bg-gray-100 rounded-md mb-2 flex items-center justify-center">
                          <Image className="w-8 h-8 text-gray-400" />
                        </div>
                        <p className="text-sm text-center">Template {template}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="captions">
              <Card>
                <CardHeader>
                  <CardTitle>Caption Generator</CardTitle>
                  <CardDescription>Generate and edit video captions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-4">
                        <Type className="w-5 h-5" />
                        <span className="font-medium">Generated Captions</span>
                      </div>
                      <div className="bg-gray-50 p-4 rounded-md">
                        <p className="text-gray-600">Captions will appear here after processing...</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
};

export default Dashboard;