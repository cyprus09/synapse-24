import { Button } from "@/components/ui/button";

const Footer = () => {
  return (
    <footer className="bg-black text-white py-6">
      <div className="container mx-auto flex flex-col md:flex-row justify-between items-center">
        <div className="flex flex-col mb-4 md:mb-0">
          <h3 className="font-semibold text-lg">Lifelong Learning@EEE</h3>
          <p className="text-sm">Today: {new Date().toLocaleDateString()}</p>
          <br />
          <p>Made with ❤️ by Mayank Pallai</p>
        </div>

        <div className="flex flex-col md:flex-row items-center">
          <ul className="flex space-x-4 mb-4 mr-3 md:mb-0">
            <li>
              <a href="#home" className="hover:underline">
                Home
              </a>
            </li>
            <li>
              <a href="#faqs" className="hover:underline">
                FAQs
              </a>
            </li>
          </ul>

          <div className="flex space-x-2">
            <Button variant="outline" size="icon" className="text-white">
              <span>🔗</span>
            </Button>
            <Button variant="outline" size="icon" className="text-white">
              <span>📱</span>
            </Button>
            <Button variant="outline" size="icon" className="text-white">
              <span>✉️</span>
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;